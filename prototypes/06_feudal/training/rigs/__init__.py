"""
Training rigs.

Rule: a module's rig is the dependency closure of its Obs contract.

  - every field synthesisable without world_v1 physics -> standalone rig
  - any field produced by sim dynamics               -> in-sim rig, oracles in other slots

  pathfinder   to_goal only                            -> pathfinder_rig_v1  (standalone)
  eat / drink  brightness, h, s, tile levels           -> insim_rig_v1
  orchestrator h, s, known flags, tile levels          -> insim_rig_v1
  explorer     legal moves, smell history, need flags  -> insim_rig_v1
"""
