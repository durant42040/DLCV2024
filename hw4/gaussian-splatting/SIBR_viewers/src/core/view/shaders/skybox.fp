/*
 * Copyright (C) 2020, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */



#version 420

layout(binding = 0) uniform samplerCube in_CubeMap;
layout(location= 0) out vec4 out_Color;

in VSOUT
{
  vec3 tc;
} in_Frag;

void main(void)
{
  out_Color = texture(in_CubeMap, in_Frag.tc);
}
