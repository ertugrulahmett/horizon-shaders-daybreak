// The lit colour of each point of a mob, a player, an armour stand or a block entity (chests,
// banners, bells), before Horizon's alpha test, the enchantment glint and fog. Lit by the same
// sun, shadow map, sky and block light as terrain (lighting.wgsl), so a sheep in a tree's
// shade is as dark as the grass it stands on. 0.2 did this in the composite, from a terrain
// mask every block had to write; the entity stage needs neither.
override relight_entities: bool = true;

// Vanilla's own, for what Daybreak leaves alone (`EntityInput` in Horizon's reference): the
// hurt flash laid over the surface in vanilla's gamma, then vanilla's light.
fn vanilla_entity(input: EntityInput) -> Surface {
    let base = input.albedo.rgb * input.tint.rgb * input.shade;
    let flashed = linear_color(mix(display_color(base), display_color(input.overlay.rgb), input.overlay.a));
    return Surface(vec4<f32>(flashed * input.lightmap, input.albedo.a * input.tint.a));
}

fn entity(input: EntityInput) -> Surface {
    // Entities (0), block entities (1) and dropped or framed items (2) once Horizon draws them
    // here. Particles and the hand keep vanilla's light, and so do glowing layers (eyes, the
    // warden's glow) and the Nether and the End, where terrain is close to vanilla too.
    let lit_kind = input.kind <= 2u;
    if !relight_entities || !lit_kind || input.emission > 0.5 || frame.dimension != 0 {
        return vanilla_entity(input);
    }
    let albedo = display_color(input.albedo.rgb);
    let tint = display_color(input.tint.rgb);
    let colored = display_color(input.colored_light);
    // The flash replaces part of the surface before the light, as vanilla's does.
    let surface = mix(albedo * tint, display_color(input.overlay.rgb), input.overlay.a);

    let n = input.normal;
    let l = frame.light_direction;
    let facing = dot(n, l);
    // As on terrain: the terminator eased in, nothing past it.
    let ndl = max(facing, 0.0) * smoothstep(0.0, 0.2, facing);
    var direct = vec3<f32>(0.0);
    if ndl > 0.0 && sun_strength > 0.0 {
        let open = smoothstep(0.0, 0.4, input.light.y);
        direct = direct_light() * (SUN_INTENSITY * sun_strength) * ndl * open
            * sun_shadow(input.position, n, facing);
    }
    let light = ambient_light(input.position, n, input.light, colored) + direct;
    var color = surface * light;
    // Night vision shows everything as if lit.
    color = mix(color, surface, frame.night_vision * 0.85);
    return Surface(vec4<f32>(linear_color(shoulder(color)), input.albedo.a * input.tint.a));
}
