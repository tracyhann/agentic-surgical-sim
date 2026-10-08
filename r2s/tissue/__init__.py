"""Per-tissue models for the 4D study (outputs/iter/). One module per tissue, each owned by one modelling track:

  gallbladder   BodyParts3D template fitted to the multi-view observations, then per frame (4D)
  membrane      the peritoneal sheet the grasper lifts into a tent
  ducts         cystic duct and pink strands as tubes
  backdrop      liver and the static surroundings, fused over the multi-view cameras
  interaction   instrument tips / jaws / shafts in 3D; what the grasper holds, where the probe pushes in

Contract: outputs/iter/CONTRACT.md.
"""
