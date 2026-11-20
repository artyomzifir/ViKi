"""One predeclared feedback pilot preserving commanded versus actual palm offset.

The existing generator normally commands an actual-palm-derived absolute pose.
This pilot instead applies its small measured object correction to the preceding
command pose, retaining the displacement that creates servo contact preload.
All physics, feasibility checks, source prefix and release gates are unchanged.
"""
import sys
from experiments import hocap_shadow_feedback_placement as feedback


def main():
    fixed = '--fixed-offset' in sys.argv
    if fixed: sys.argv.remove('--fixed-offset')
    feedback.SETTINGS = ({'name': 'firm_preserve_command_deflection',
                          'object_step_m': .003, 'object_turn_deg': 1.5,
                          'palm_accel_m': .0015, 'palm_angular_accel_deg': .75,
                          'fixed_command_deflection': fixed,
                          'preserve_command_deflection': not fixed},)
    feedback.PROTOCOL = {**feedback.PROTOCOL,
        'schema': 'shadow_preload_placement_protocol_v1', 'candidate_count': 1,
        'settings': feedback.SETTINGS,
        'hypothesis': ('preserve fixed actual-to-command transform measured at110 without integration windup' if fixed else 'apply the measured world SE3 correction to the preceding command, preserving servo preload'),
        'controlled_touchdown': 'independently require pre-support triad>=90%, max gap<=.05s, impact speed<.1m/s; upright landing alone is not success'}
    feedback.main()


if __name__ == '__main__': main()
