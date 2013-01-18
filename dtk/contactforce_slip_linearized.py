from numpy import sin, cos, tan, arctan, pi, sqrt
import numpy as np

# v here is from metadata['Speed'] of each run.
# Here the contact points slip rate are not obtained yet from taskSignals,
# but, we can integrate the contact points acceleration.

def contact_force_slip_linearized(v, mooreParameters, taskSignals):
    """Return contact forces of each wheel under the slip condition
    after linearizing the nonlinear slip model.

    v : float
        The forward speed for the specific task. It is the operation points of
        the linearized slip model.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fx_r_s, Fy_r_s, Fx_f_s, Fy_f_s : float
        The contact forces of each wheel, rear wheel longitudinal, rear wheel
        lateral, front wheel longitudinal, front wheel lateral, respectively,
        under the slip condition. All are expressed by individual body-fixed 
        coordinates of the wheel

    """
    f0 = np.vectorize(contact_force_rear_longitudinal_slip_linearized)
    Fx_r_s = f0(mooreParameters, taskSignals)

    f1 = np.vectorize(contact_force_rear_lateral_slip_linearized)
    Fy_r_s = f1(v, mooreParameters, taskSignals)

    f2 = np.vectorize(contact_force_front_longitudinal_slip_linearized)
    Fx_f_s = f2(mooreParameters, taskSignals)

    f3 = np.vectorize(contact_force_front_lateral_slip_linearized)
    Fy_f_s = f3(v, mooreParameters, taskSignals)

    return Fx_r_s, Fy_r_s, Fx_f_s, Fy_f_s

def contact_force_rear_longitudinal_slip_linearized(mooreParameters, taskSignals):

    """Return longitudinal contact force of rear wheel under the slip condition
    after linearizing the nonlinear unconstraint model.

    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fx_r_s : float
        The rear wheel longitudinal contact force along A['1'] direction 
        under the slip condition.

    """
    mp = mooreParameters
    ts = taskSignals

    rF = mp['rf']; rR = mp['rr']
    d1 = mp['d1']; d2 = mp['d2']; d3 = mp['d3']
    l1 = mp['l1']; l2 = mp['l2']; l3 = mp['l3']; l4 = mp['l4']
    mc = mp['mc']; md = mp['md']; me = mp['me']; mf = mp['mf']
    ic22 = mp['ic22']; id22 = mp['id22']; ie22 = mp['ie22']; if22 = mp['if22']

    u3 = ts['PitchRate']; u5 = ts['RearWheelRate']; u6 = ts['FrontWheelRate']
    u7 = ts['LongRearConRate']; u9 = ts['LongFrontConRate']

    u3d = ts['PitchAcc']; u5d = ts['RearWheelAcc']; u6d = ts['FrontWheelAcc']
    u7d = ts['LongRearConAcc']; u9d = ts['LongFrontConAcc']

    Fx_r_s = -u3d*(mc*(0.312*l1 - 0.95*l2 + rR) + md*rR - ((0.312*d1 + \
        0.312*d3 - 0.0975*rF + 0.0975*rR - (d2 + 0.95*rF - \
        0.95*rR)*(0.312*d1 - 0.95*d2 + 0.312*d3 - 1.0*rF + rR))/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2) + \
        (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF \
        + 0.312*rR))*(0.95*rF*(0.312*d1 - 0.95*d2 + 0.312*d3 - 1.0*rF + \
        rR) + 0.0975*rF + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))*(0.312*d1 + 0.312*d3 - \
        0.0975*rF + 0.0975*rR - (d2 + 0.95*rF - 0.95*rR)*(0.312*d1 - \
        0.95*d2 + 0.312*d3 - 1.0*rF + rR))/((d2 + 0.95*rF - 0.95*rR)**2 + \
        (d1 + d3 - 0.312*rF + 0.312*rR)**2))/((1.0*rF**2 - (0.95*rF*(d2 + \
        0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))**2/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF \
        + 0.312*rR)**2))*((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2)))*(ic22 + id22 + ie22 + if22 + mc*(l1**2 \
        + 0.312*l1*rR + l2**2 - 0.95*l2*rR + rR**2 + rR*(0.312*l1 - \
        0.95*l2)) + md*rR**2 + me*((l3 + 0.312*rF)**2 + (l4 - \
        0.95*rF)**2) + 1.0*mf*rF**2) + (if22 + me*(0.312*rF*(l3 + \
        0.312*rF) - 0.95*rF*(l4 - 0.95*rF)) + \
        1.0*mf*rF**2)*(0.95*rF*(0.312*d1 - 0.95*d2 + 0.312*d3 - 1.0*rF + \
        rR) + 0.0975*rF + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))*(0.312*d1 + 0.312*d3 - \
        0.0975*rF + 0.0975*rR - (d2 + 0.95*rF - 0.95*rR)*(0.312*d1 - \
        0.95*d2 + 0.312*d3 - 1.0*rF + rR))/((d2 + 0.95*rF - 0.95*rR)**2 + \
        (d1 + d3 - 0.312*rF + 0.312*rR)**2))/(1.0*rF**2 - (0.95*rF*(d2 + \
        0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))**2/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF \
        + 0.312*rR)**2))) - u5d*(mc*rR + md*rR - ((0.312*d1 + 0.312*d3 - \
        0.0975*rF + 0.0975*rR - (d2 + 0.95*rF - 0.95*rR)*(0.312*d1 - \
        0.95*d2 + 0.312*d3 - 1.0*rF + rR))/((d2 + 0.95*rF - 0.95*rR)**2 + \
        (d1 + d3 - 0.312*rF + 0.312*rR)**2) + (0.95*rF*(d2 + 0.95*rF - \
        0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))*(0.95*rF*(0.312*d1 - 0.95*d2 + 0.312*d3 - 1.0*rF + rR) \
        + 0.0975*rF + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + \
        d3 - 0.312*rF + 0.312*rR))*(0.312*d1 + 0.312*d3 - 0.0975*rF + \
        0.0975*rR - (d2 + 0.95*rF - 0.95*rR)*(0.312*d1 - 0.95*d2 + \
        0.312*d3 - 1.0*rF + rR))/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 \
        - 0.312*rF + 0.312*rR)**2))/((1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF \
        - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*((d2 \
        + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2)))*(id22 + mc*(0.312*l1*rR - 0.95*l2*rR + rR**2) + \
        md*rR**2)) - u6d*(-((0.312*d1 + 0.312*d3 - 0.0975*rF + 0.0975*rR \
        - (d2 + 0.95*rF - 0.95*rR)*(0.312*d1 - 0.95*d2 + 0.312*d3 - \
        1.0*rF + rR))/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF \
        + 0.312*rR)**2) + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))*(0.95*rF*(0.312*d1 - \
        0.95*d2 + 0.312*d3 - 1.0*rF + rR) + 0.0975*rF + (0.95*rF*(d2 + \
        0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))*(0.312*d1 + 0.312*d3 - 0.0975*rF + 0.0975*rR - (d2 + \
        0.95*rF - 0.95*rR)*(0.312*d1 - 0.95*d2 + 0.312*d3 - 1.0*rF + \
        rR))/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))/((1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*((d2 + 0.95*rF \
        - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2)))*(if22 + \
        me*(0.312*rF*(l3 + 0.312*rF) - 0.95*rF*(l4 - 0.95*rF)) + \
        1.0*mf*rF**2) + (if22 + 1.0*me*rF**2 + \
        1.0*mf*rF**2)*(0.95*rF*(0.312*d1 - 0.95*d2 + 0.312*d3 - 1.0*rF + \
        rR) + 0.0975*rF + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))*(0.312*d1 + 0.312*d3 - \
        0.0975*rF + 0.0975*rR - (d2 + 0.95*rF - 0.95*rR)*(0.312*d1 - \
        0.95*d2 + 0.312*d3 - 1.0*rF + rR))/((d2 + 0.95*rF - 0.95*rR)**2 + \
        (d1 + d3 - 0.312*rF + 0.312*rR)**2))/(1.0*rF**2 - (0.95*rF*(d2 + \
        0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))**2/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF \
        + 0.312*rR)**2))) + u7d*(mc + md - (mc*(0.312*l1 - 0.95*l2 + rR) \
        + md*rR)*((0.312*d1 + 0.312*d3 - 0.0975*rF + 0.0975*rR - (d2 + \
        0.95*rF - 0.95*rR)*(0.312*d1 - 0.95*d2 + 0.312*d3 - 1.0*rF + \
        rR))/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2) + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 \
        + d3 - 0.312*rF + 0.312*rR))*(0.95*rF*(0.312*d1 - 0.95*d2 + \
        0.312*d3 - 1.0*rF + rR) + 0.0975*rF + (0.95*rF*(d2 + 0.95*rF - \
        0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))*(0.312*d1 + \
        0.312*d3 - 0.0975*rF + 0.0975*rR - (d2 + 0.95*rF - \
        0.95*rR)*(0.312*d1 - 0.95*d2 + 0.312*d3 - 1.0*rF + rR))/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))/((1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*((d2 + 0.95*rF \
        - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2)))) + \
        u9d*(-(me*(0.312*l3 - 0.95*l4 + 1.0*rF) + 1.0*mf*rF)*((0.312*d1 + \
        0.312*d3 - 0.0975*rF + 0.0975*rR - (d2 + 0.95*rF - \
        0.95*rR)*(0.312*d1 - 0.95*d2 + 0.312*d3 - 1.0*rF + rR))/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2) + \
        (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF \
        + 0.312*rR))*(0.95*rF*(0.312*d1 - 0.95*d2 + 0.312*d3 - 1.0*rF + \
        rR) + 0.0975*rF + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))*(0.312*d1 + 0.312*d3 - \
        0.0975*rF + 0.0975*rR - (d2 + 0.95*rF - 0.95*rR)*(0.312*d1 - \
        0.95*d2 + 0.312*d3 - 1.0*rF + rR))/((d2 + 0.95*rF - 0.95*rR)**2 + \
        (d1 + d3 - 0.312*rF + 0.312*rR)**2))/((1.0*rF**2 - (0.95*rF*(d2 + \
        0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))**2/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF \
        + 0.312*rR)**2))*((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2))) + (1.0*me*rF + \
        1.0*mf*rF)*(0.95*rF*(0.312*d1 - 0.95*d2 + 0.312*d3 - 1.0*rF + rR) \
        + 0.0975*rF + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + \
        d3 - 0.312*rF + 0.312*rR))*(0.312*d1 + 0.312*d3 - 0.0975*rF + \
        0.0975*rR - (d2 + 0.95*rF - 0.95*rR)*(0.312*d1 - 0.95*d2 + \
        0.312*d3 - 1.0*rF + rR))/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 \
        - 0.312*rF + 0.312*rR)**2))/(1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - \
        0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2)))

    return Fx_r_s

def contact_force_rear_lateral_slip_linearized(v, mooreParameters, taskSignals):

    """Return lateral contact force of rear wheel under the slip condition
    after linearizing the nonlinear unconstraint model.

    v : float
        The forward speed for the specific task. It is the operation points of
        the linearized slip model.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fy_r_s : float
        The rear wheel lateral contact force along A['2'] direction 
        under the slip condition.

    """
    mp = mooreParameters
    ts = taskSignals

    rF = mp['rf']; rR = mp['rr']
    d1 = mp['d1']; d2 = mp['d2']; d3 = mp['d3']
    l1 = mp['l1']; l2 = mp['l2']; l3 = mp['l3']; l4 = mp['l4']; g = mp['g']
    mc = mp['mc']; md = mp['md']; me = mp['me']; mf = mp['mf']
    ic11 = mp['ic11']; ic22 = mp['ic22']; ic33 = mp['ic33']; ic31 = mp['ic31']
    id11 = mp['id11']; id22 = mp['id22']; if11 = mp['if11']; if22 = mp['if22']
    ie11 = mp['ie11']; ie22 = mp['ie22']; ie33 = mp['ie33']; ie31 = mp['ie31']

    q2 = ts['RollAngle']; q4 = ts['SteerAngle']

    u1 = ts['YawRate']; u2 = ts['RollRate']; u3 = ts['PitchRate']
    u4 = ts['SteerRate']; u5 = ts['RearWheelRate']; u6 = ts['FrontWheelRate']
    u7 = ts['LongRearConRate']; u8 = ts['LatRearConRate']
    u9 = ts['LongFrontConRate']; u10 = ts['LatFrontConRate']

    u1d = ts['YawAcc']; u2d = ts['RollAcc']; u3d = ts['PitchAcc']
    u4d= ts['SteerAcc']; u5d = ts['RearWheelAcc']; u6d = ts['FrontWheelAcc']
    u7d = ts['LongRearConAcc']; u8d = ts['LatRearConAcc']
    u9d = ts['LongFrontConAcc']; u10d = ts['LatFrontConAcc']

    Fy_r_s = -me*u10d*(0.95*l3 + 0.312*l4)/(0.95*d1 + 0.312*d2 + 0.95*d3) \
        - q2*(((0.95*d1 + 0.95*d3 - 0.297*rF + 0.297*rR - ((d2 + 0.95*rF \
        - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2)/(0.95*d1 + \
        0.312*d2 + 0.95*d3))/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2) + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))*(0.297*rF + \
        (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF \
        + 0.312*rR))/(0.95*d1 + 0.312*d2 + 0.95*d3) + (0.95*rF*(d2 + \
        0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))*(0.95*d1 + 0.95*d3 - 0.297*rF + 0.297*rR - ((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2)/(0.95*d1 + 0.312*d2 + 0.95*d3))/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))/((1.0*rF**2 - \
        (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF \
        + 0.312*rR))**2/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2))*((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 \
        - 0.312*rF + 0.312*rR)**2)))*(0.95*g*l1*mc + 0.312*g*l2*mc + \
        0.95*g*me*(l3 + 0.312*rF) + 0.312*g*me*(l4 - 0.95*rF)) + \
        (0.95*g*l1*mc + 0.312*g*l2*mc - g*mc*(0.95*l1 + 0.312*l2) - \
        g*me*(0.95*l3 + 0.312*l4) + 0.95*g*me*(l3 + 0.312*rF) + \
        0.312*g*me*(l4 - 0.95*rF))/(0.95*d1 + 0.312*d2 + 0.95*d3)) - \
        q4*(-0.95*v*(mc*v + md*v - (mc*v*(0.95*l1 + 0.312*l2) + \
        1.0*me*v*(0.95*l3 + 0.312*l4))/(0.95*d1 + 0.312*d2 + \
        0.95*d3))/(0.95*d1 + 0.312*d2 + 0.95*d3) - ((d2 + 0.95*rF - \
        0.95*rR + ((0.312*d3 - 0.0975*rF)*(d1 + d3 - 0.312*rF + 0.312*rR) \
        - (0.95*d1 + 0.312*d2 + 0.95*d3)*(d2 + 0.95*rF - 0.95*rR) + \
        (0.95*d1 + 0.312*d2 + 0.297*rF)*(d2 + 0.95*rF - \
        0.95*rR))/(0.95*d1 + 0.312*d2 + 0.95*d3))/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2) + (0.95*rF*(d2 \
        + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))*((0.312*rF*(0.312*d3 - 0.0975*rF) - 0.95*rF*(0.95*d1 + \
        0.312*d2 + 0.297*rF))/(0.95*d1 + 0.312*d2 + 0.95*d3) + \
        (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF \
        + 0.312*rR))*(d2 + 0.95*rF - 0.95*rR + ((0.312*d3 - \
        0.0975*rF)*(d1 + d3 - 0.312*rF + 0.312*rR) - (0.95*d1 + 0.312*d2 \
        + 0.95*d3)*(d2 + 0.95*rF - 0.95*rR) + (0.95*d1 + 0.312*d2 + \
        0.297*rF)*(d2 + 0.95*rF - 0.95*rR))/(0.95*d1 + 0.312*d2 + \
        0.95*d3))/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))/((1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*((d2 + 0.95*rF \
        - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2)))*(0.95*g*l1*mc + 0.312*g*l2*mc + 0.95*g*me*(l3 + \
        0.312*rF) + 0.312*g*me*(l4 - 0.95*rF)) + (0.95*g*me*(0.312*l3 + \
        0.0975*rF) - 0.312*g*me*(0.95*l3 + 0.312*l4) + \
        0.312*g*me*(0.312*l4 - 0.297*rF))/(0.95*d1 + 0.312*d2 + 0.95*d3)) \
        - u10*(0.297*d1 + 0.0975*d2 + 0.297*d3)*(mc*v + md*v - \
        (mc*v*(0.95*l1 + 0.312*l2) + 1.0*me*v*(0.95*l3 + \
        0.312*l4))/(0.95*d1 + 0.312*d2 + 0.95*d3))/(0.95*d1 + 0.312*d2 + \
        0.95*d3)**2 + u1d*(mc*(0.95*l1 + 0.312*l2) - (0.0975*ic11 - \
        0.593*ic31 + 0.903*ic33 + 1.0*id11 + 0.0975*ie11 - 0.593*ie31 + \
        0.903*ie33 + 1.0*if11 + mc*(0.95*l1 + 0.312*l2)**2 + me*(0.95*l3 \
        + 0.312*l4)**2)/(0.95*d1 + 0.312*d2 + 0.95*d3)) - \
        u2*(-4.16e-17*id11*v/rR - 1.0*id22*v/rR - 1.0*if22*v/rF + (mc*v + \
        md*v - (mc*v*(0.95*l1 + 0.312*l2) + 1.0*me*v*(0.95*l3 + \
        0.312*l4))/(0.95*d1 + 0.312*d2 + 0.95*d3))*(0.312*d1 - 0.95*d2 + \
        0.312*d3 - 1.0*rF + rR))/(0.95*d1 + 0.312*d2 + 0.95*d3) + \
        u2d*(mc*(0.312*l1 - 0.95*l2 + rR) + md*rR - (-0.297*ic11 + \
        0.805*ic31 + 0.297*ic33 - 0.297*ie11 + 0.805*ie31 + 0.297*ie33 + \
        mc*(rR*(0.95*l1 + 0.312*l2) + (0.312*l1 - 0.95*l2)*(0.95*l1 + \
        0.312*l2)) + me*(0.95*l3 + 0.312*l4)*(0.312*l3 - 0.95*l4 + \
        1.0*rF))/(0.95*d1 + 0.312*d2 + 0.95*d3)) - u3d*(((d3 - \
        0.312*rF)*(d2 + 0.95*rF - 0.95*rR)/((d2 + 0.95*rF - 0.95*rR)**2 + \
        (d1 + d3 - 0.312*rF + 0.312*rR)**2) - (0.95*rF*(d3 - 0.312*rF) - \
        (d3 - 0.312*rF)*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 \
        + d3 - 0.312*rF + 0.312*rR))*(d2 + 0.95*rF - 0.95*rR)/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + \
        d3 - 0.312*rF + 0.312*rR))/((1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - \
        0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*((d2 \
        + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2)))*(ic22 + id22 + ie22 + if22 + mc*(l1**2 + \
        0.312*l1*rR + l2**2 - 0.95*l2*rR + rR**2 + rR*(0.312*l1 - \
        0.95*l2)) + md*rR**2 + me*((l3 + 0.312*rF)**2 + (l4 - \
        0.95*rF)**2) + 1.0*mf*rF**2) + (0.95*rF*(d3 - 0.312*rF) - (d3 - \
        0.312*rF)*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - \
        0.312*rF + 0.312*rR))*(d2 + 0.95*rF - 0.95*rR)/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*(if22 + \
        me*(0.312*rF*(l3 + 0.312*rF) - 0.95*rF*(l4 - 0.95*rF)) + \
        1.0*mf*rF**2)/(1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))) - \
        u4*(-0.312*if22*v/rF + 0.95*me*v*(0.95*l3 + 0.312*l4) + (d3 - \
        0.312*rF)*(mc*v + md*v - (mc*v*(0.95*l1 + 0.312*l2) + \
        1.0*me*v*(0.95*l3 + 0.312*l4))/(0.95*d1 + 0.312*d2 + \
        0.95*d3)))/(0.95*d1 + 0.312*d2 + 0.95*d3) - u4d*(-0.312*ie31 + \
        0.95*ie33 + 0.95*if11 + me*(0.95*l3 + 0.312*l4)*(l3 + \
        0.312*rF))/(0.95*d1 + 0.312*d2 + 0.95*d3) - u5d*((d3 - \
        0.312*rF)*(d2 + 0.95*rF - 0.95*rR) - (0.95*rF*(d3 - 0.312*rF) - \
        (d3 - 0.312*rF)*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 \
        + d3 - 0.312*rF + 0.312*rR))*(d2 + 0.95*rF - 0.95*rR)/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + \
        d3 - 0.312*rF + 0.312*rR))/(1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - \
        0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2)))*(id22 + mc*(0.312*l1*rR - 0.95*l2*rR + rR**2) + \
        md*rR**2)/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2) - u6d*(((d3 - 0.312*rF)*(d2 + 0.95*rF - \
        0.95*rR)/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2) - (0.95*rF*(d3 - 0.312*rF) - (d3 - \
        0.312*rF)*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - \
        0.312*rF + 0.312*rR))*(d2 + 0.95*rF - 0.95*rR)/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*(0.95*rF*(d2 + \
        0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))/((1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*((d2 + 0.95*rF \
        - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2)))*(if22 + \
        me*(0.312*rF*(l3 + 0.312*rF) - 0.95*rF*(l4 - 0.95*rF)) + \
        1.0*mf*rF**2) + (0.95*rF*(d3 - 0.312*rF) - (d3 - \
        0.312*rF)*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - \
        0.312*rF + 0.312*rR))*(d2 + 0.95*rF - 0.95*rR)/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*(if22 + \
        1.0*me*rF**2 + 1.0*mf*rF**2)/(1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF \
        - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))) + \
        u7d*(mc*(0.312*l1 - 0.95*l2 + rR) + md*rR)*((d3 - 0.312*rF)*(d2 + \
        0.95*rF - 0.95*rR) - (0.95*rF*(d3 - 0.312*rF) - (d3 - \
        0.312*rF)*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - \
        0.312*rF + 0.312*rR))*(d2 + 0.95*rF - 0.95*rR)/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*(0.95*rF*(d2 + \
        0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))/(1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2)))/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2) - \
        u8*(mc*v + md*v - (mc*v*(0.95*l1 + 0.312*l2) + 1.0*me*v*(0.95*l3 \
        + 0.312*l4))/(0.95*d1 + 0.312*d2 + 0.95*d3))/(0.95*d1 + 0.312*d2 \
        + 0.95*d3) + u8d*(-mc*(0.95*l1 + 0.312*l2)/(0.95*d1 + 0.312*d2 + \
        0.95*d3) + mc + md) + u9d*((me*(0.312*l3 - 0.95*l4 + 1.0*rF) + \
        1.0*mf*rF)*((d3 - 0.312*rF)*(d2 + 0.95*rF - 0.95*rR)/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2) - \
        (0.95*rF*(d3 - 0.312*rF) - (d3 - 0.312*rF)*(0.95*rF*(d2 + 0.95*rF \
        - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))*(d2 + \
        0.95*rF - 0.95*rR)/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2))*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))/((1.0*rF**2 - \
        (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF \
        + 0.312*rR))**2/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2))*((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 \
        - 0.312*rF + 0.312*rR)**2))) + (1.0*me*rF + \
        1.0*mf*rF)*(0.95*rF*(d3 - 0.312*rF) - (d3 - \
        0.312*rF)*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - \
        0.312*rF + 0.312*rR))*(d2 + 0.95*rF - 0.95*rR)/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))/(1.0*rF**2 - \
        (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF \
        + 0.312*rR))**2/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2))) 

    return Fy_r_s

def contact_force_front_longitudinal_slip_linearized(mooreParameters, taskSignals):

    """Return longitudinal contact force of front wheel under the slip condition
    after linearizing the nonlinear unconstraint model.

    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fx_f_s : float
        The front wheel longitudinal contact force along longitudinal direction 
        of front wheel under the slip condition.

    """
    mp = mooreParameters
    ts = taskSignals

    rF = mp['rf']; rR = mp['rr']
    l1 = mp['l1']; l2 = mp['l2']; l3 = mp['l3']; l4 = mp['l4']; g = mp['g']
    d1 = mp['d1']; d2 = mp['d2']; d3 = mp['d3']
    mc = mp['mc']; md = mp['md']; me = mp['me']; mf = mp['mf']
    ic11 = mp['ic11']; ic22 = mp['ic22']; ic33 = mp['ic33']; ic31 = mp['ic31']
    id11 = mp['id11']; id22 = mp['id22']; if11 = mp['if11']; if22 = mp['if22']
    ie11 = mp['ie11']; ie22 = mp['ie22']; ie33 = mp['ie33']; ie31 = mp['ie31']

    q4 = ts['SteerAngle']

    u3 = ts['PitchRate']; u5 = ts['RearWheelRate']; u6 = ts['FrontWheelRate']
    u7 = ts['LongRearConRate']; u9 = ts['LongFrontConRate']

    u3d = ts['PitchAcc']; u5d = ts['RearWheelAcc']; u6d = ts['FrontWheelAcc']
    u7d = ts['LongRearConAcc']; u9d = ts['LongFrontConAcc']

    Fx_f_s = -g*q4*(0.95*rR*(d2 + 0.95*rF - 0.95*rR) - (0.903*rF*rR - \
        0.95*rR*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - \
        0.312*rF + 0.312*rR))*(d2 + 0.95*rF - 0.95*rR)/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*(0.95*rF*(d2 + \
        0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))/(1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2)))*(0.95*l1*mc + \
        0.312*l2*mc + 0.95*me*(l3 + 0.312*rF) + 0.312*me*(l4 - \
        0.95*rF))/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2) - u3d*(me*(0.312*l3 - 0.95*l4 + 1.0*rF) + 1.0*mf*rF \
        - (0.0975*rF + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + \
        d3 - 0.312*rF + 0.312*rR))*(0.312*d1 + 0.312*d3 - 0.0975*rF + \
        0.0975*rR)/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))*(if22 + me*(0.312*rF*(l3 + 0.312*rF) - 0.95*rF*(l4 \
        - 0.95*rF)) + 1.0*mf*rF**2)/(1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - \
        0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2)) + \
        ((0.0975*rF + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + \
        d3 - 0.312*rF + 0.312*rR))*(0.312*d1 + 0.312*d3 - 0.0975*rF + \
        0.0975*rR)/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + \
        d3 - 0.312*rF + 0.312*rR))/((1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - \
        0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*((d2 \
        + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2)) + \
        (0.312*d1 + 0.312*d3 - 0.0975*rF + 0.0975*rR)/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*(ic22 + id22 + \
        ie22 + if22 + mc*(l1**2 + 0.312*l1*rR + l2**2 - 0.95*l2*rR + \
        rR**2 + rR*(0.312*l1 - 0.95*l2)) + md*rR**2 + me*((l3 + \
        0.312*rF)**2 + (l4 - 0.95*rF)**2) + 1.0*mf*rF**2)) - u5d*(id22 + \
        mc*(0.312*l1*rR - 0.95*l2*rR + rR**2) + md*rR**2)*(0.312*d1 + \
        0.312*d3 - 0.0975*rF + 0.0975*rR + (0.0975*rF + (0.95*rF*(d2 + \
        0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))*(0.312*d1 + 0.312*d3 - 0.0975*rF + 0.0975*rR)/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + \
        d3 - 0.312*rF + 0.312*rR))/(1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - \
        0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2)))/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2) - u6d*(1.0*me*rF + 1.0*mf*rF - \
        (0.0975*rF + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + \
        d3 - 0.312*rF + 0.312*rR))*(0.312*d1 + 0.312*d3 - 0.0975*rF + \
        0.0975*rR)/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))*(if22 + 1.0*me*rF**2 + 1.0*mf*rF**2)/(1.0*rF**2 - \
        (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF \
        + 0.312*rR))**2/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2)) + ((0.0975*rF + (0.95*rF*(d2 + 0.95*rF \
        - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))*(0.312*d1 \
        + 0.312*d3 - 0.0975*rF + 0.0975*rR)/((d2 + 0.95*rF - 0.95*rR)**2 \
        + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*(0.95*rF*(d2 + 0.95*rF - \
        0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))/((1.0*rF**2 \
        - (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - \
        0.312*rF + 0.312*rR))**2/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 \
        - 0.312*rF + 0.312*rR)**2))*((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + \
        d3 - 0.312*rF + 0.312*rR)**2)) + (0.312*d1 + 0.312*d3 - 0.0975*rF \
        + 0.0975*rR)/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))*(if22 + me*(0.312*rF*(l3 + 0.312*rF) - 0.95*rF*(l4 \
        - 0.95*rF)) + 1.0*mf*rF**2)) + u7d*(mc*(0.312*l1 - 0.95*l2 + rR) \
        + md*rR)*(0.312*d1 + 0.312*d3 - 0.0975*rF + 0.0975*rR + \
        (0.0975*rF + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + \
        d3 - 0.312*rF + 0.312*rR))*(0.312*d1 + 0.312*d3 - 0.0975*rF + \
        0.0975*rR)/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + \
        d3 - 0.312*rF + 0.312*rR))/(1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - \
        0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2)))/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2) + u9d*(me + mf - (0.0975*rF + \
        (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF \
        + 0.312*rR))*(0.312*d1 + 0.312*d3 - 0.0975*rF + 0.0975*rR)/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))*(1.0*me*rF + 1.0*mf*rF)/(1.0*rF**2 - (0.95*rF*(d2 \
        + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))**2/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF \
        + 0.312*rR)**2)) + (me*(0.312*l3 - 0.95*l4 + 1.0*rF) + \
        1.0*mf*rF)*((0.0975*rF + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))*(0.312*d1 + 0.312*d3 - \
        0.0975*rF + 0.0975*rR)/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2))*(0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))/((1.0*rF**2 - \
        (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF \
        + 0.312*rR))**2/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2))*((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 \
        - 0.312*rF + 0.312*rR)**2)) + (0.312*d1 + 0.312*d3 - 0.0975*rF + \
        0.0975*rR)/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2)))

    return Fx_f_s
 
def contact_force_front_lateral_slip_linearized(v, mooreParameters, taskSignals):

    """Return lateral contact force of front wheel under the slip condition
    after linearizing the nonlinear unconstraint model.

    v : float
        The forward speed for the specific task. It is the operation points of
        the linearized slip model.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fy_f_s : float
        The front wheel lateral contact force along lateral direction of front
        wheel under the slip condition.

    """
    mp = mooreParameters
    ts = taskSignals

    rF = mp['rf']; rR = mp['rr']
    l1 = mp['l1']; l2 = mp['l2']; l3 = mp['l3']; l4 = mp['l4']; g = mp['g']
    d1 = mp['d1']; d2 = mp['d2']; d3 = mp['d3']
    mc = mp['mc']; md = mp['md']; me = mp['me']; mf = mp['mf']
    ic11 = mp['ic11']; ic33 = mp['ic33']; ic31 = mp['ic31']
    id11 = mp['id11']; id22 = mp['id22']; if11 = mp['if11']; if22 = mp['if22']
    ie11 = mp['ie11']; ie33 = mp['ie33']; ie31 = mp['ie31']

    q2 = ts['RollAngle']; q4 = ts['SteerAngle']

    u1 = ts['YawRate']; u2 = ts['RollRate']; u4 = ts['SteerRate']
    u8 = ts['LatRearConRate']; u10 = ts['LatFrontConRate']

    u1d = ts['YawAcc']; u2d = ts['RollAcc']; u4d= ts['SteerAcc'] 
    u8d = ts['LatRearConAcc']; u10d = ts['LatFrontConAcc']

    Fy_f_s = -mc*u8d*(0.95*l1 + 0.312*l2)*(0.297*d1 + 0.0975*d2 + \
        0.297*d3)/(0.95*d1 + 0.312*d2 + 0.95*d3)**2 - q2*((-(0.95*d1 + \
        0.95*d3 - 0.297*rF + 0.297*rR + ((d2 + 0.95*rF - 0.95*rR)**2 + \
        (d1 + d3 - 0.312*rF + 0.312*rR)**2)*(0.297*d1 + 0.0975*d2 + \
        0.297*d3)/(0.95*d1 + 0.312*d2 + 0.95*d3)**2)/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2) - (0.95*rF*(d2 \
        + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))*(0.297*rF - (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))*(0.297*d1 + 0.0975*d2 + \
        0.297*d3)/(0.95*d1 + 0.312*d2 + 0.95*d3)**2 + (0.95*rF*(d2 + \
        0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + \
        0.312*rR))*(0.95*d1 + 0.95*d3 - 0.297*rF + 0.297*rR + ((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2)*(0.297*d1 + 0.0975*d2 + 0.297*d3)/(0.95*d1 + \
        0.312*d2 + 0.95*d3)**2)/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - \
        0.312*rF + 0.312*rR)**2))/((1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - \
        0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*((d2 \
        + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2)))*(0.95*g*l1*mc + 0.312*g*l2*mc + 0.95*g*me*(l3 + \
        0.312*rF) + 0.312*g*me*(l4 - 0.95*rF)) + (0.297*d1 + 0.0975*d2 + \
        0.297*d3)*(0.95*g*l1*mc + 0.312*g*l2*mc - g*mc*(0.95*l1 + \
        0.312*l2) - g*me*(0.95*l3 + 0.312*l4) + 0.95*g*me*(l3 + 0.312*rF) \
        + 0.312*g*me*(l4 - 0.95*rF))/(0.95*d1 + 0.312*d2 + 0.95*d3)**2) - \
        q4*(-0.95*v*(1.0*me*v + 1.0*mf*v - (mc*v*(0.95*l1 + 0.312*l2) + \
        1.0*me*v*(0.95*l3 + 0.312*l4))*(0.297*d1 + 0.0975*d2 + \
        0.297*d3)/(0.95*d1 + 0.312*d2 + 0.95*d3)**2)/(0.95*d1 + 0.312*d2 \
        + 0.95*d3) + (-(0.297*d1 + 0.312*d2 + 0.297*d3 + 0.204*rF - \
        0.204*rR - (0.95*d3 - 0.297*rF)*(d2 + 0.95*rF - 0.95*rR) + \
        (0.297*d1 + 0.0975*d2 + 0.297*d3)*((0.312*d3 - 0.0975*rF)*(d1 + \
        d3 - 0.312*rF + 0.312*rR) - (0.95*d1 + 0.312*d2 + 0.95*d3)*(d2 + \
        0.95*rF - 0.95*rR) + (0.95*d1 + 0.312*d2 + 0.297*rF)*(d2 + \
        0.95*rF - 0.95*rR))/(0.95*d1 + 0.312*d2 + 0.95*d3)**2)/((d2 + \
        0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2) - \
        (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 + d3 - 0.312*rF \
        + 0.312*rR))*(0.95*rF*(0.95*d3 - 0.297*rF) + 0.0926*rF + \
        (0.312*rF*(0.312*d3 - 0.0975*rF) - 0.95*rF*(0.95*d1 + 0.312*d2 + \
        0.297*rF))*(0.297*d1 + 0.0975*d2 + 0.297*d3)/(0.95*d1 + 0.312*d2 \
        + 0.95*d3)**2 + (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - 0.312*rF*(d1 \
        + d3 - 0.312*rF + 0.312*rR))*(0.297*d1 + 0.312*d2 + 0.297*d3 + \
        0.204*rF - 0.204*rR - (0.95*d3 - 0.297*rF)*(d2 + 0.95*rF - \
        0.95*rR) + (0.297*d1 + 0.0975*d2 + 0.297*d3)*((0.312*d3 - \
        0.0975*rF)*(d1 + d3 - 0.312*rF + 0.312*rR) - (0.95*d1 + 0.312*d2 \
        + 0.95*d3)*(d2 + 0.95*rF - 0.95*rR) + (0.95*d1 + 0.312*d2 + \
        0.297*rF)*(d2 + 0.95*rF - 0.95*rR))/(0.95*d1 + 0.312*d2 + \
        0.95*d3)**2)/((d2 + 0.95*rF - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2))/((1.0*rF**2 - (0.95*rF*(d2 + 0.95*rF - 0.95*rR) - \
        0.312*rF*(d1 + d3 - 0.312*rF + 0.312*rR))**2/((d2 + 0.95*rF - \
        0.95*rR)**2 + (d1 + d3 - 0.312*rF + 0.312*rR)**2))*((d2 + 0.95*rF \
        - 0.95*rR)**2 + (d1 + d3 - 0.312*rF + \
        0.312*rR)**2)))*(0.95*g*l1*mc + 0.312*g*l2*mc + 0.95*g*me*(l3 + \
        0.312*rF) + 0.312*g*me*(l4 - 0.95*rF)) + (0.297*d1 + 0.0975*d2 + \
        0.297*d3)*(0.95*g*me*(0.312*l3 + 0.0975*rF) - 0.312*g*me*(0.95*l3 \
        + 0.312*l4) + 0.312*g*me*(0.312*l4 - 0.297*rF))/(0.95*d1 + \
        0.312*d2 + 0.95*d3)**2) - u10*(0.297*d1 + 0.0975*d2 + \
        0.297*d3)*(1.0*me*v + 1.0*mf*v - (mc*v*(0.95*l1 + 0.312*l2) + \
        1.0*me*v*(0.95*l3 + 0.312*l4))*(0.297*d1 + 0.0975*d2 + \
        0.297*d3)/(0.95*d1 + 0.312*d2 + 0.95*d3)**2)/(0.95*d1 + 0.312*d2 \
        + 0.95*d3)**2 + u10d*(-me*(0.95*l3 + 0.312*l4)*(0.297*d1 + \
        0.0975*d2 + 0.297*d3)/(0.95*d1 + 0.312*d2 + 0.95*d3)**2 + me + \
        mf) + u1d*(me*(0.95*l3 + 0.312*l4) - (0.297*d1 + 0.0975*d2 + \
        0.297*d3)*(0.0975*ic11 - 0.593*ic31 + 0.903*ic33 + 1.0*id11 + \
        0.0975*ie11 - 0.593*ie31 + 0.903*ie33 + 1.0*if11 + mc*(0.95*l1 + \
        0.312*l2)**2 + me*(0.95*l3 + 0.312*l4)**2)/(0.95*d1 + 0.312*d2 + \
        0.95*d3)**2) + u2*((0.297*d1 + 0.0975*d2 + \
        0.297*d3)*(4.16e-17*id11*v/rR + 1.0*id22*v/rR + \
        1.0*if22*v/rF)/(0.95*d1 + 0.312*d2 + 0.95*d3) - (1.0*me*v + \
        1.0*mf*v - (mc*v*(0.95*l1 + 0.312*l2) + 1.0*me*v*(0.95*l3 + \
        0.312*l4))*(0.297*d1 + 0.0975*d2 + 0.297*d3)/(0.95*d1 + 0.312*d2 \
        + 0.95*d3)**2)*(0.312*d1 - 0.95*d2 + 0.312*d3 - 1.0*rF + \
        rR))/(0.95*d1 + 0.312*d2 + 0.95*d3) + u2d*(me*(0.312*l3 - 0.95*l4 \
        + 1.0*rF) + 1.0*mf*rF - (0.297*d1 + 0.0975*d2 + \
        0.297*d3)*(-0.297*ic11 + 0.805*ic31 + 0.297*ic33 - 0.297*ie11 + \
        0.805*ie31 + 0.297*ie33 + mc*(rR*(0.95*l1 + 0.312*l2) + (0.312*l1 \
        - 0.95*l2)*(0.95*l1 + 0.312*l2)) + me*(0.95*l3 + \
        0.312*l4)*(0.312*l3 - 0.95*l4 + 1.0*rF))/(0.95*d1 + 0.312*d2 + \
        0.95*d3)**2) + u4*(0.95*me*v + 0.95*mf*v - (d3 - \
        0.312*rF)*(1.0*me*v + 1.0*mf*v - (mc*v*(0.95*l1 + 0.312*l2) + \
        1.0*me*v*(0.95*l3 + 0.312*l4))*(0.297*d1 + 0.0975*d2 + \
        0.297*d3)/(0.95*d1 + 0.312*d2 + 0.95*d3)**2)/(0.95*d1 + 0.312*d2 \
        + 0.95*d3) + (0.312*if22*v/rF - 0.95*me*v*(0.95*l3 + \
        0.312*l4))*(0.297*d1 + 0.0975*d2 + 0.297*d3)/(0.95*d1 + 0.312*d2 \
        + 0.95*d3)**2) + u4d*(me*(l3 + 0.312*rF) + 0.312*mf*rF - \
        (0.297*d1 + 0.0975*d2 + 0.297*d3)*(-0.312*ie31 + 0.95*ie33 + \
        0.95*if11 + me*(0.95*l3 + 0.312*l4)*(l3 + 0.312*rF))/(0.95*d1 + \
        0.312*d2 + 0.95*d3)**2) - u8*(1.0*me*v + 1.0*mf*v - \
        (mc*v*(0.95*l1 + 0.312*l2) + 1.0*me*v*(0.95*l3 + \
        0.312*l4))*(0.297*d1 + 0.0975*d2 + 0.297*d3)/(0.95*d1 + 0.312*d2 \
        + 0.95*d3)**2)/(0.95*d1 + 0.312*d2 + 0.95*d3) 

    return Fy_f_s
