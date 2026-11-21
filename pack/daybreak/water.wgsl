// Water's surface for the material stage: waves, reflection, refraction and the sun's glint.

// Precision-safe 2D hash returning 0..1 without large-coordinate sine artifacts.
fn water_hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + vec3<f32>(33.33));
    return fract((p3.x + p3.y) * p3.z);
}

// Analytical gradient of quintic-interpolated 2D value noise, smooth across cell borders.
fn water_noise_grad(p: vec2<f32>) -> vec2<f32> {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    let a = water_hash(i);
    let b = water_hash(i + vec2<f32>(1.0, 0.0));
    let c = water_hash(i + vec2<f32>(0.0, 1.0));
    let d = water_hash(i + vec2<f32>(1.0, 1.0));

    let k1 = b - a;
    let k2 = c - a;
    let k3 = a - b - c + d;
    return du * (vec2<f32>(k1, k2) + k3 * u.yx);
}

fn rotate2(v: vec2<f32>, cs: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(v.x * cs.x - v.y * cs.y, v.x * cs.y + v.y * cs.x);
}

fn water_ripple(xz: vec2<f32>, scale: f32, cs: vec2<f32>, shift: vec2<f32>) -> vec2<f32> {
    let g = water_noise_grad(rotate2(xz, cs) * scale + shift);
    return rotate2(g, vec2<f32>(cs.x, -cs.y));
}

// The wave's normal at world `xz`, combining analytical swell slopes with filtered multi-octave
// ripples. Each octave fades smoothly with the projected pixel footprint so distant water
// never aliases into Moiré rings.
fn water_normal(xz: vec2<f32>, t: f32, dist: f32, view_y: f32) -> vec3<f32> {
    if !water_waves {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    let footprint = dist * 0.0018 / max(abs(view_y), 0.05);
    var slope = vec2<f32>(0.0);

    let a0 = footprint * 0.9;
    let f0 = exp(-(a0 * a0));
    if f0 > 0.002 {
        let swell_grad = vec2<f32>(0.42, 0.25) * (cos(dot(xz, vec2<f32>(0.42, 0.25)) + t * 1.1) * 0.225)
            + vec2<f32>(-0.31, 0.58) * (cos(dot(xz, vec2<f32>(-0.31, 0.58)) + t * 1.45) * 0.175)
            + vec2<f32>(0.67, -0.52) * (cos(dot(xz, vec2<f32>(0.67, -0.52)) - t * 1.85) * 0.10);
        slope -= swell_grad * (WATER_AMPLITUDE * 1.2 * f0);
    }

    let a1 = footprint * 2.2;
    let f1 = exp(-(a1 * a1));
    if f1 > 0.002 {
        slope -= water_ripple(xz, 1.8, vec2<f32>(0.8, 0.6), vec2<f32>(t * 0.55, -t * 0.35)) * (0.045 * f1);
    }

    let a2 = footprint * 4.5;
    let f2 = exp(-(a2 * a2));
    if f2 > 0.002 {
        slope -= water_ripple(xz, 3.7, vec2<f32>(0.28, -0.96), vec2<f32>(-t * 0.45, t * 0.70)) * (0.032 * f2);
    }

    let a3 = footprint * 9.0;
    let f3 = exp(-(a3 * a3));
    if f3 > 0.002 {
        slope -= water_ripple(xz, 7.5, vec2<f32>(-0.8, 0.6), vec2<f32>(t * 0.80, t * 0.65)) * (0.016 * f3);
    }

    return normalize(vec3<f32>(slope.x, 1.0, slope.y));
}

// The distance from the camera of what lies at screen point `screen` with reversed `depth`.
fn distance_at(screen: vec2<f32>, depth: f32) -> f32 {
    if depth <= 0.0 {
        return 1e4;
    }
    let clip = vec4<f32>(screen.x * 2.0 - 1.0, 1.0 - screen.y * 2.0, depth, 1.0);
    let view = frame.projection_inverse * clip;
    return length(view.xyz / view.w);
}

// Screen-space reflection: the reflected ray `r` from camera-relative `p`, marched across the
// world's picture before translucent terrain (`opaque_color` and `opaque_depth`). It costs no
// pass of its own: the mirror draws the terrain a second time, which also keeps Horizon from
// culling the world on the GPU. What the screen doesn't show (above its top, behind the
// camera) is the sky's colour, `fallback`. Twelve steps, each longer than the last, out to
// about 100 blocks, then four halvings around the first one that passes just behind something.
const SSR_STEPS: i32 = 12;
const SSR_REFINE: i32 = 4;

fn screen_of(q: vec3<f32>) -> vec3<f32> {
    let clip = frame.projection * (frame.view * vec4<f32>(q, 1.0));
    if clip.w <= 0.01 {
        return vec3<f32>(-1.0);
    }
    let ndc = clip.xyz / clip.w;
    return vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

fn screen_reflection(p: vec3<f32>, r: vec3<f32>, jitter: f32, fallback: vec3<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(opaque_color));
    var t_prev = 0.0;
    var t = 0.6 * (0.6 + 0.8 * jitter);
    for (var i = 0; i < SSR_STEPS; i++) {
        let q = p + r * t;
        let s = screen_of(q);
        if any(s.xy < vec2<f32>(0.0)) || any(s.xy > vec2<f32>(1.0)) {
            break;
        }
        let texel = vec2<i32>(min(s.xy * size, size - 1.0));
        let scene = textureLoad(opaque_depth, texel, 0);
        // Depth is reversed: the scene is nearer than the ray's point, so the ray went behind it.
        if scene > s.z {
            // Close enough behind to be what it hit, not something far in front of it.
            let gap = length(q) - distance_at(s.xy, scene);
            if gap < max(1.0, t * 0.35) {
                var lo = t_prev;
                var hi = t;
                for (var j = 0; j < SSR_REFINE; j++) {
                    let mid = 0.5 * (lo + hi);
                    let m = screen_of(p + r * mid);
                    let mt = vec2<i32>(clamp(m.xy, vec2<f32>(0.0), vec2<f32>(1.0)) * (size - 1.0));
                    if textureLoad(opaque_depth, mt, 0) > m.z {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let h = screen_of(p + r * hi);
                let uv = clamp(h.xy, vec2<f32>(0.0), vec2<f32>(1.0));
                let hit = textureLoad(opaque_color, vec2<i32>(uv * (size - 1.0)), 0).rgb;
                // Fade out toward the screen's edges, where the picture runs out.
                let edge = min(min(uv.x, 1.0 - uv.x) * 8.0, min(uv.y, 1.0 - uv.y) * 6.0);
                return mix(fallback, hit, clamp(edge, 0.0, 1.0));
            }
            // Far behind: the ray passed behind something in front of what it reflects (a
            // tree before the shore). It carries on.
        }
        t_prev = t;
        t *= 1.55;
    }
    return fallback;
}

fn water_surface(input: MaterialInput) -> vec4<f32> {
    let world = input.position + frame.camera_position;
    let dist = length(input.position);
    let to_eye = normalize(-input.position);
    // The surface (also a flowing slope, and seen from below), by its normal: reading `face`
    // would make terrain take a third again its memory.
    let top = abs(input.normal.y) > 0.5;
    var n = input.normal;
    if top {
        n = water_normal(world.xz, frame.time, dist, to_eye.y);
        // Seen from below, the surface faces down.
        n = select(n, -n, input.normal.y < 0.0);
    }
    // What lies behind the surface, a little bent by the waves.
    let behind_size = vec2<f32>(textureDimensions(opaque_color));
    let bent = clamp(input.screen + n.xz * 0.02, vec2<f32>(0.0), vec2<f32>(1.0));
    let texel = vec2<i32>(min(bent * behind_size, behind_size - 1.0));
    let behind = display_color(textureLoad(opaque_color, texel, 0).rgb);
    let behind_distance = distance_at(bent, textureLoad(opaque_depth, texel, 0));
    let thickness = max(behind_distance - dist, 0.0);

    // The water's own colour (the biome's tint over its texture) in the light around it.
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let ambient = ambient_light(input.position, up, input.light, display_color(input.colored_light));
    let sun = direct_light() * sun_strength * sun_shadow(input.position, up, frame.light_direction.y);
    let body = display_color(input.albedo.rgb) * display_color(input.shaded_tint.rgb) * (ambient + sun * 0.4);
    // Light is absorbed red first, then green: deep water turns blue, then the water's colour.
    let reach = mix(2.0, 14.0, water_clarity);
    let absorb = exp(-thickness / reach * vec3<f32>(2.2, 0.9, 0.6));
    var color = display_color(mix(linear_color(body), linear_color(behind), absorb));

    // Underwater, the surface overhead is mostly a window.
    if frame.eye_in == 1u {
        return vec4<f32>(mix(linear_color(color), linear_color(body), 0.25), 1.0);
    }

    // Reflection, by Schlick's Fresnel (water's F0 is 0.02).
    let cos_v = clamp(dot(n, to_eye), 0.0, 1.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - cos_v, 5.0);
    // Covered water (little sky light) reflects little of the sky.
    let amount = fresnel * smoothstep(0.2, 0.8, input.light.y);
    let r = reflect(-to_eye, n);
    // The sky in the reflected direction: vanilla's fog at the horizon, its sky colour above.
    // Reflections are mixed in linear light, as `opaque_color` and `reflection` hold it.
    let sky_mix = mix(display_color(frame.fog_color), display_color(frame.sky_color), smoothstep(0.0, 0.35, r.y));
    var mirrored = linear_color(sky_mix * mix(0.15, 1.0, daylight()));
    if reflections && top && frame.has_reflection == 1u {
        let uv = clamp(input.screen + n.xz * 0.03, vec2<f32>(0.0), vec2<f32>(1.0));
        mirrored = textureSampleLevel(reflection, reflection_sampler, uv, 0.0).rgb;
    } else if screen_reflections && top && amount > 0.02 {
        // Looking almost straight down, the reflection is a few percent: not worth the march.
        mirrored = screen_reflection(input.position, r, ign(input.screen * frame.resolution), mirrored);
    }
    color = display_color(mix(linear_color(color), mirrored, amount));

    // The sun's glint.
    let h = normalize(frame.light_direction + to_eye);
    let glint = pow(max(dot(n, h), 0.0), 400.0) * 4.0;
    color += sun * glint;
    return vec4<f32>(linear_color(shoulder(color)), 1.0);
}
