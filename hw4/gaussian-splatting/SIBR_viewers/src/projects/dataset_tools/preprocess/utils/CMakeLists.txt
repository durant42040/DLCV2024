# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


project(dataset_tools_utils)

file(GLOB SCRIPTS "*.py")

add_custom_target(${PROJECT_NAME} ALL)

include(install_runtime)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "projects/dataset_tools/preprocess")
ibr_install_rsc(${PROJECT_NAME} TYPE "scripts" FOLDER "utils" FILES ${SCRIPTS})
