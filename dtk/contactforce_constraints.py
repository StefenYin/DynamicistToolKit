from numpy import sin, cos, tan, arctan, pi, sqrt
import numpy as np


def contact_force_constraints(lam, mooreParameters, taskSignals):

    f0 = np.vectorize(contact_force_rear_longitudinal_constraints)
    Fx_r_c = f0(lam, mooreParameters, taskSignals)

    f1 = np.vectorize(contact_force_rear_lateral_constraints)
    Fy_r_c = f1(lam, mooreParameters, taskSignals)

    f2 = np.vectorize(contact_force_rear_normal_constraints)
    Fz_r_c = f2(lam, mooreParameters, taskSignals)

    f3 = np.vectorize(contact_force_front_longitudinal_constraints)
    Fx_f_c = f3(lam, mooreParameters, taskSignals)

    f4 = np.vectorize(contact_force_front_lateral_constraints)
    Fy_f_c = f4(lam, mooreParameters, taskSignals)

    f5 = np.vectorize(contact_force_front_normal_constraints)
    Fz_f_c = f5(lam, mooreParameters, taskSignals)

    return Fx_r_c, Fy_r_c, Fz_r_c, Fx_f_c, Fy_f_c, Fz_f_c


def contact_force_rear_longitudinal_constraints(lam, mooreParameters, taskSignals):

    """Return longitudinal contact force of rear wheel under the constraint condition.

    lam : float
        The tilt angle.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fx_r_c : float
        The rear wheel longitudinal contact force along forward direction of 
        the intersection line between rear wheel plane and horizontal plane 
        under the constraint condition.

    """

    mp = mooreParameters
    ts = taskSignals

    d1 = mp['d1']; d2 = mp['d2']; d3 = mp['d3']
    l1 = mp['l1']; l2 = mp['l2']; l3 = mp['l3']; l4 = mp['l4']
    rr = mp['rr']; rf = mp['rf']; g = mp['g']
    mc = mp['mc'];  md = mp['md']; me = mp['me'];  mf = mp['mf']
    ic11 = mp['ic11']; ic22 = mp['ic22']; ic33 = mp['ic33']; ic31 = mp['ic31']
    id11 = mp['id11']; id22 = mp['id22']; if11 = mp['if11']; if22 = mp['if22']
    ie11 = mp['ie11']; ie22 = mp['ie22']; ie33 = mp['ie33']; ie31 = mp['ie31']

    q2 = ts['RollAngle']; q3 = lam; q4 = ts['SteerAngle']
    u1 = ts['YawRate'];  u2 = ts['RollRate']; u3 = ts['PitchRate']
    u4 = ts['SteerRate']; u5 = ts['RearWheelRate']; u6 = ts['FrontWheelRate']
    u1d = ts['YawAcc']; u2d = ts['RollAcc']; u3d = ts['PitchAcc']
    u4d = ts['SteerAcc']; u5d = ts['RearWheelAcc']; u6d = ts['FrontWheelAcc']

    Fx_r_c = -(mc*(l1*u1**2*cos(q3) + l1*u1*u2*sin(q3)**3*cos(q2) + \
        l1*u1*u2*sin(q3)*cos(q2)*cos(q3)**2 + l1*u1*u2*sin(q3)*cos(q2) + \
        2.0*l1*u1*u3*sin(q2)*cos(q3) + l1*u3**2*cos(q3) + \
        l1*sin(q2)*sin(q3)*u1d + l1*sin(q3)*u3d + \
        l2*u1**2*sin(q2)**2*sin(q3) + l2*u1**2*sin(q3)**3*cos(q2)**2 + \
        l2*u1**2*sin(q3)*cos(q2)**2*cos(q3)**2 - \
        2.0*l2*u1*u2*cos(q2)*cos(q3) + 2.0*l2*u1*u3*sin(q2)*sin(q3) + \
        l2*u3**2*sin(q3) - l2*sin(q2)*cos(q3)*u1d - l2*cos(q3)*u3d + \
        rr*u1*u2*sin(q3)**4*cos(q2) - rr*u1*u2*cos(q2)*cos(q3)**4 + \
        2.0*rr*u1*u2*cos(q2)*cos(q3)**2 + rr*u1*u2*cos(q2) + \
        rr*sin(q2)*u1d + rr*u3d + rr*u5d) + \
        md*rr*(u1*u2*sin(q3)**4*cos(q2) - u1*u2*cos(q2)*cos(q3)**4 + \
        2.0*u1*u2*cos(q2)*cos(q3)**2 + u1*u2*cos(q2) + sin(q2)*u1d + u3d \
        + u5d) + (g*mc*((-l1 - rr*sin(q3))*cos(q2)*cos(q3) - (l2 - \
        rr*cos(q3))*sin(q3)*cos(q2)) + g*me*((-l3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4))*cos(q2)*cos(q3) + \
        (-l4*sin(q4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 \
        + cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3))*(sin(q2 \
        )*cos(q4) + sin(q3)*sin(q4)*cos(q2)) + (l4*cos(q4) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2* \
        cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))) + g*mf*rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*sin( \
        q4)*cos(q2)*cos(q3))*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*cos(q3)*cos(q4) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*sin(q3)*cos(q2)*cos(q3) + \
        (rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + rr \
        *sin(q2)*sin(q3))*cos(q2)*cos(q3))*sin(q4)*cos(q3))/(-rf*((rf*(( \
        sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (-((-rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) - (d2 - \
        rr*cos(q3))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3)) + (-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3)))*cos(q3)*cos(q4) - ((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*sin(q3) + (-(rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3)) - (-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3)))*sin(q4)*cos(q3))*(-if22*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) - me*rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        me*rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3) - mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q3)*cos(q4) - \
        rf*(rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q3)*cos(q2)*cos(q3) + \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) - (d2 - \
        rr*cos(q3))*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + rr*sin(q3))*cos(q2)*cos( \
        q3))*sin(q4)*cos(q3))*(-id22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q2) - if22*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) - mc*(-l1*sin(q2) - \
        rr*sin(q2)*sin(q3))*(-l1*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        l2*(u1*sin(q2) + u3)**2 + rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3))) - \
        mc*(l2*sin(q2) - rr*sin(q2)*cos(q3))*(-l1*(u1*sin(q2) + u3)**2 + \
        l2*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*sin(q3) + rr*(u1*sin(q2) + u3 + \
        u5)*u3*sin(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*cos(q3) - (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) - (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3))) - mc*(l1*cos(q2)*cos(q3) + \
        l2*sin(q3)*cos(q2))*(l1*(u1*sin(q2) + u3)*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + l1*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) \
        + u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        l2*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - u2*u3*sin(q3) \
        - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + rr*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*(u1*sin(q2) + u3 + u5)*sin(q3) - \
        rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*u3*sin(q3) - \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*(u1*sin(q2) + u3 + \
        u5)*cos(q3) + rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*u3*cos(q3) + \
        rr*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - u2*u3*sin(q3) \
        - sin(q3)*cos(q2)*u1d + cos(q3)*u2d)*cos(q3) + \
        rr*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d)*sin(q3)) + \
        md*rr*(-rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + u5)*sin(q3) + \
        rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - rr*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d + u5d)*cos(q3) - (rr*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*cos(q3) + rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)))*sin(q2)*cos(q3) + md*rr*(rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + \
        u5)*u3*cos(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q3) + (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*sin(q2)*sin(q3) - me*(-l3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*(l4*(sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2)) - \
        rf*(sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)* \
        cos(q2)*cos(q3))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - me*(l3*cos(q2)*cos(q3) - l4*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4)))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-if11*(u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + u4)*u6 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + if22*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-ie11*((u1*sin(q2) + u3)*u4*cos(q4) + \
        (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + ie22*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        ie31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - \
        (ie31*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + ie33*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4))) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-ie22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3) + u4) + (ie31*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie33*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        (-ic22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) + (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*sin(q2) + (id11*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*(u1*sin(q2) + u3 + u5) - id11*((-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u5 - u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) - \
        id22*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*(u1*sin(q2) + u3 + \
        u5))*cos(q2)*cos(q3) - (-id11*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*(u1*sin(q2) + u3 + u5) - id11*(-(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*u5 + u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + \
        id22*(u1*cos(q2)*cos(q3) + u2*sin(q3))*(u1*sin(q2) + u3 + \
        u5))*sin(q3)*cos(q2) + (if11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*(((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))*u6 \
        - u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + \
        sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - if22*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6))*cos(q2)*cos(q3) - \
        (-ic11*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + \
        ic22*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        ic31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) - \
        (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*sin(q2) + \
        u3))*sin(q3)*cos(q2) + (-ic22*(u1*sin(q2) + \
        u3)*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) - \
        ic31*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) - \
        ic33*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*sin(q2) + \
        u3))*cos(q2)*cos(q3) + (-ie22*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) - ie31*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - ie33*(-u1*u2*sin(q2)*cos(q3) \
        - u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4)))*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*cos(q3)*cos(q4) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*sin(q3)*cos(q2)*cos(q3) + \
        (rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + rr \
        *sin(q2)*sin(q3))*cos(q2)*cos(q3))*sin(q4)*cos(q3))*(-ic22*(u1*u2 \
        *cos(q2) + sin(q2)*u1d + u3d) - id22*(u1*u2*cos(q2) + sin(q2)*u1d \
        + u3d + u5d) - if22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q4) - mc*(-l1 - \
        rr*sin(q3))*(-l1*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        l2*(u1*sin(q2) + u3)**2 + rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3))) - mc*(l2 \
        - rr*cos(q3))*(-l1*(u1*sin(q2) + u3)**2 + l2*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*sin(q3) + rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q3) - \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3))) + md*rr*(-rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*sin(q3) + rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q3) - \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3)))*cos(q3) + md*rr*(rr*(u1*sin(q2) + u3)*(u1*sin(q2) \
        + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*sin(q3) \
        - me*(-l3*cos(q4) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*(-l4*sin(q4) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*sin( \
        q4)*cos(q2)*cos(q3))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        me*(l4*cos(q4) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*cos( \
        q2)*cos(q3)*cos(q4))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q4) + mf*rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3)*cos(q4) - mf*rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + \
        cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3))*sin(q4)*cos(q2)*cos(q3) - \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) + (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + (-ie22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3) + u4) + (ie31*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie33*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q4) + (-if11*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + u4)*u6 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + if22*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4) + u6))*sin(q4) + \
        (-ie11*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + \
        ie22*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4) - ie31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - \
        (ie31*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + ie33*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + \
        u3*cos(q4)))*sin(q4))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)))

    return Fx_r_c


def contact_force_rear_lateral_constraints(lam, mooreParameters, taskSignals):

    """Return lateral contact force of rear wheel under the constraint condition.

    lam : float
        The tilt angle.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fy_r_c : float
        The rear wheel lateral contact force along lateral direction, perpendicular
        to Fx_r_c and pointing to the right side of a rider, under the constraint 
        condition.

    """

    mp = mooreParameters
    ts = taskSignals

    d1 = mp['d1']; d2 = mp['d2']; d3 = mp['d3']
    l1 = mp['l1']; l2 = mp['l2']; l3 = mp['l3']; l4 = mp['l4']
    rr = mp['rr']; rf = mp['rf']; g = mp['g']
    mc = mp['mc'];  md = mp['md']; me = mp['me'];  mf = mp['mf']
    ic11 = mp['ic11']; ic22 = mp['ic22']; ic33 = mp['ic33']; ic31 = mp['ic31']
    id11 = mp['id11']; id22 = mp['id22']; if11 = mp['if11']; if22 = mp['if22']
    ie11 = mp['ie11']; ie22 = mp['ie22']; ie33 = mp['ie33']; ie31 = mp['ie31']

    q2 = ts['RollAngle']; q3 = lam; q4 = ts['SteerAngle']
    u1 = ts['YawRate'];  u2 = ts['RollRate']; u3 = ts['PitchRate']
    u4 = ts['SteerRate']; u5 = ts['RearWheelRate']; u6 = ts['FrontWheelRate']
    u1d = ts['YawAcc']; u2d = ts['RollAcc']; u3d = ts['PitchAcc']
    u4d = ts['SteerAcc']; u5d = ts['RearWheelAcc']; u6d = ts['FrontWheelAcc']

    Fy_r_c = -(mc*(l1*u1**2*sin(q2)**3*sin(q3) + l1*u1**2*sin(q2)*sin(q3)*cos(q2)**2 \
        + 2.0*l1*u1*u3*sin(q3) + l1*u2**2*sin(q2)*sin(q3)**3 + \
        l1*u2**2*sin(q2)*sin(q3)*cos(q3)**2 - \
        2.0*l1*u2*u3*cos(q2)*cos(q3) + l1*u3**2*sin(q2)*sin(q3) - \
        l1*sin(q2)*cos(q3)*u3d - l1*sin(q3)*cos(q2)*u2d - l1*cos(q3)*u1d \
        - l2*u1**2*sin(q2)**3*cos(q3) - \
        l2*u1**2*sin(q2)*cos(q2)**2*cos(q3) + \
        l2*u1*u2*sin(q2)*sin(q3)**3*cos(q2) + \
        l2*u1*u2*sin(q2)*sin(q3)*cos(q2)*cos(q3)**2 - \
        l2*u1*u2*sin(q2)*sin(q3)*cos(q2) - 2.0*l2*u1*u3*cos(q3) - \
        l2*u2**2*sin(q2)*cos(q3) - 2.0*l2*u2*u3*sin(q3)*cos(q2) - \
        l2*u3**2*sin(q2)*cos(q3) - l2*sin(q2)*sin(q3)*u3d - \
        l2*sin(q3)*u1d + l2*cos(q2)*cos(q3)*u2d + rr*u1**2*sin(q2)**3 + \
        rr*u1**2*sin(q2)*cos(q2)**2 - rr*u1*u3*sin(q2)**2*cos(q3)**2 + \
        rr*u1*u3*sin(q2)**2 - rr*u1*u3*cos(q2)**2*cos(q3)**2 + \
        rr*u1*u3*cos(q2)**2 + rr*u1*u3*cos(q3)**2 - \
        rr*u1*u5*sin(q2)**2*cos(q3)**2 + rr*u1*u5*sin(q2)**2 - \
        rr*u1*u5*cos(q2)**2*cos(q3)**2 + rr*u1*u5*cos(q2)**2 + \
        rr*u1*u5*cos(q3)**2 + rr*u2**2*sin(q2)*sin(q3)**4 - \
        rr*u2**2*sin(q2)*cos(q3)**4 + 2.0*rr*u2**2*sin(q2)*cos(q3)**2 - \
        rr*cos(q2)*u2d) + md*rr*(u1**2*sin(q2)**3 + \
        u1**2*sin(q2)*cos(q2)**2 - u1*u3*sin(q2)**2*cos(q3)**2 + \
        u1*u3*sin(q2)**2 - u1*u3*cos(q2)**2*cos(q3)**2 + u1*u3*cos(q2)**2 \
        + u1*u3*cos(q3)**2 - u1*u5*sin(q2)**2*cos(q3)**2 + \
        u1*u5*sin(q2)**2 - u1*u5*cos(q2)**2*cos(q3)**2 + u1*u5*cos(q2)**2 \
        + u1*u5*cos(q3)**2 + u2**2*sin(q2)*sin(q3)**4 - \
        u2**2*sin(q2)*cos(q3)**4 + 2.0*u2**2*sin(q2)*cos(q3)**2 - \
        cos(q2)*u2d) + (-((-rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) - (d2 - \
        rr*cos(q3))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3)) + (-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3)))*(sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2)) + \
        ((rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*sin(q2)*cos(q3) - \
        (-(rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3)) - (-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3)))*(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4)))*(-if22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) - me*rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        me*rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3) - mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + (g*mc*((-l1 - \
        rr*sin(q3))*cos(q2)*cos(q3) - (l2 - rr*cos(q3))*sin(q3)*cos(q2)) \
        + g*me*((-l3*cos(q4) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4))*cos(q2)*cos(q3) + \
        (-l4*sin(q4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 \
        + cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3))*(sin(q2 \
        )*cos(q4) + sin(q3)*sin(q4)*cos(q2)) + (l4*cos(q4) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2* \
        cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))) + g*mf*rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*sin( \
        q4)*cos(q2)*cos(q3))*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 \
        + cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + \
        (d2*sin(q2) - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*sin(q2)*cos(q2)*cos(q3)**2 - \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*(rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3)))/(-rf*((rf*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*(rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rf*(rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q2)*cos(q2)*cos(q3)**2 - \
        (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*(-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) - (d2 - \
        rr*cos(q3))*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3)))*(-id22*(u1*u2*cos(q2) + sin(q2)*u1d \
        + u3d + u5d)*sin(q2) - if22*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) - mc*(-l1*sin(q2) - \
        rr*sin(q2)*sin(q3))*(-l1*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        l2*(u1*sin(q2) + u3)**2 + rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3))) - \
        mc*(l2*sin(q2) - rr*sin(q2)*cos(q3))*(-l1*(u1*sin(q2) + u3)**2 + \
        l2*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*sin(q3) + rr*(u1*sin(q2) + u3 + \
        u5)*u3*sin(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*cos(q3) - (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) - (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3))) - mc*(l1*cos(q2)*cos(q3) + \
        l2*sin(q3)*cos(q2))*(l1*(u1*sin(q2) + u3)*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + l1*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) \
        + u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        l2*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - u2*u3*sin(q3) \
        - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + rr*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*(u1*sin(q2) + u3 + u5)*sin(q3) - \
        rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*u3*sin(q3) - \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*(u1*sin(q2) + u3 + \
        u5)*cos(q3) + rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*u3*cos(q3) + \
        rr*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - u2*u3*sin(q3) \
        - sin(q3)*cos(q2)*u1d + cos(q3)*u2d)*cos(q3) + \
        rr*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d)*sin(q3)) + \
        md*rr*(-rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + u5)*sin(q3) + \
        rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - rr*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d + u5d)*cos(q3) - (rr*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*cos(q3) + rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)))*sin(q2)*cos(q3) + md*rr*(rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + \
        u5)*u3*cos(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q3) + (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*sin(q2)*sin(q3) - me*(-l3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*(l4*(sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2)) - \
        rf*(sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)* \
        cos(q2)*cos(q3))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - me*(l3*cos(q2)*cos(q3) - l4*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4)))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-if11*(u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + u4)*u6 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + if22*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-ie11*((u1*sin(q2) + u3)*u4*cos(q4) + \
        (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + ie22*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        ie31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - \
        (ie31*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + ie33*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4))) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-ie22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3) + u4) + (ie31*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie33*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        (-ic22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) + (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*sin(q2) + (id11*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*(u1*sin(q2) + u3 + u5) - id11*((-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u5 - u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) - \
        id22*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*(u1*sin(q2) + u3 + \
        u5))*cos(q2)*cos(q3) - (-id11*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*(u1*sin(q2) + u3 + u5) - id11*(-(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*u5 + u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + \
        id22*(u1*cos(q2)*cos(q3) + u2*sin(q3))*(u1*sin(q2) + u3 + \
        u5))*sin(q3)*cos(q2) + (if11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*(((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))*u6 \
        - u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + \
        sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - if22*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6))*cos(q2)*cos(q3) - \
        (-ic11*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + \
        ic22*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        ic31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) - \
        (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*sin(q2) + \
        u3))*sin(q3)*cos(q2) + (-ic22*(u1*sin(q2) + \
        u3)*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) - \
        ic31*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) - \
        ic33*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*sin(q2) + \
        u3))*cos(q2)*cos(q3) + (-ie22*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) - ie31*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - ie33*(-u1*u2*sin(q2)*cos(q3) \
        - u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4)))*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 \
        + cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + \
        (d2*sin(q2) - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*sin(q2)*cos(q2)*cos(q3)**2 - \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*(rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3)))*(-ic22*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - id22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d) - if22*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d + \
        u6d)*cos(q4) - mc*(-l1 - rr*sin(q3))*(-l1*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - l2*(u1*sin(q2) + u3)**2 + rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + \
        u5)*u3*cos(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q3) + (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3))) - mc*(l2 \
        - rr*cos(q3))*(-l1*(u1*sin(q2) + u3)**2 + l2*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*sin(q3) + rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q3) - \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3))) + md*rr*(-rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*sin(q3) + rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q3) - \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3)))*cos(q3) + md*rr*(rr*(u1*sin(q2) + u3)*(u1*sin(q2) \
        + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*sin(q3) \
        - me*(-l3*cos(q4) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*(-l4*sin(q4) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*sin( \
        q4)*cos(q2)*cos(q3))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        me*(l4*cos(q4) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*cos( \
        q2)*cos(q3)*cos(q4))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q4) + mf*rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3)*cos(q4) - mf*rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + \
        cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3))*sin(q4)*cos(q2)*cos(q3) - \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) + (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + (-ie22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3) + u4) + (ie31*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie33*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q4) + (-if11*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + u4)*u6 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + if22*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4) + u6))*sin(q4) + \
        (-ie11*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + \
        ie22*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4) - ie31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - \
        (ie31*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + ie33*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + \
        u3*cos(q4)))*sin(q4))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)))

    return Fy_r_c

def contact_force_rear_normal_constraints(lam, mooreParameters, taskSignals):

    """Return normal contact force of rear wheel under the constraint condition.

    lam : float
        The tilt angle.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fz_r_c : float
        The rear wheel normal contact force along vertical direction A['3'], 
        perpendicular to the horizonal plane, under the constraint condition.

    """

    mp = mooreParameters
    ts = taskSignals

    d1 = mp['d1']; d2 = mp['d2']; d3 = mp['d3']
    l1 = mp['l1']; l2 = mp['l2']; l3 = mp['l3']; l4 = mp['l4']
    rr = mp['rr']; rf = mp['rf']; g = mp['g']
    mc = mp['mc'];  md = mp['md']; me = mp['me'];  mf = mp['mf']
    ic11 = mp['ic11']; ic22 = mp['ic22']; ic33 = mp['ic33']; ic31 = mp['ic31']
    id11 = mp['id11']; id22 = mp['id22']; if11 = mp['if11']; if22 = mp['if22']
    ie11 = mp['ie11']; ie22 = mp['ie22']; ie33 = mp['ie33']; ie31 = mp['ie31']

    q2 = ts['RollAngle']; q3 = lam; q4 = ts['SteerAngle']
    u1 = ts['YawRate'];  u2 = ts['RollRate']; u3 = ts['PitchRate']
    u4 = ts['SteerRate']; u5 = ts['RearWheelRate']; u6 = ts['FrontWheelRate']
    u1d = ts['YawAcc']; u2d = ts['RollAcc']; u3d = ts['PitchAcc']
    u4d = ts['SteerAcc']; u5d = ts['RearWheelAcc']; u6d = ts['FrontWheelAcc']

    Fz_r_c = -(g*mc + g*md + mc*(-l1*u2**2*sin(q3)**3*cos(q2) - \
        l1*u2**2*sin(q3)*cos(q2)*cos(q3)**2 - \
        2.0*l1*u2*u3*sin(q2)*cos(q3) - l1*u3**2*sin(q3)*cos(q2) - \
        l1*sin(q2)*sin(q3)*u2d + l1*cos(q2)*cos(q3)*u3d - \
        l2*u1*u2*sin(q3)**3*cos(q2)**2 - \
        l2*u1*u2*sin(q3)*cos(q2)**2*cos(q3)**2 + \
        l2*u1*u2*sin(q3)*cos(q2)**2 + l2*u2**2*cos(q2)*cos(q3) - \
        2.0*l2*u2*u3*sin(q2)*sin(q3) + l2*u3**2*cos(q2)*cos(q3) + \
        l2*sin(q2)*cos(q3)*u2d + l2*sin(q3)*cos(q2)*u3d - \
        rr*u2**2*sin(q3)**4*cos(q2) + rr*u2**2*cos(q2)*cos(q3)**4 - \
        2.0*rr*u2**2*cos(q2)*cos(q3)**2 - rr*sin(q2)*u2d) + \
        md*rr*(-u2**2*sin(q3)**4*cos(q2) + u2**2*cos(q2)*cos(q3)**4 - \
        2.0*u2**2*cos(q2)*cos(q3)**2 - sin(q2)*u2d) + \
        (-((-rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) - (d2 - \
        rr*cos(q3))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3)) + (-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3)))*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4)) - \
        ((rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*cos(q2)*cos(q3) - \
        (-(rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3)) - (-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3)))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)))*(-if22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) - me*rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        me*rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3) - mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + (g*mc*((-l1 - \
        rr*sin(q3))*cos(q2)*cos(q3) - (l2 - rr*cos(q3))*sin(q3)*cos(q2)) \
        + g*me*((-l3*cos(q4) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4))*cos(q2)*cos(q3) + \
        (-l4*sin(q4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 \
        + cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3))*(sin(q2 \
        )*cos(q4) + sin(q3)*sin(q4)*cos(q2)) + (l4*cos(q4) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2* \
        cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))) + g*mf*rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*sin( \
        q4)*cos(q2)*cos(q3))*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*cos(q2)**2*cos(q3)**2 - \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*(rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3)))/(-rf*((rf*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2*(rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*(rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)**2*cos(q3)**2 - \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*(-rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*(-rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) - (d2 - \
        rr*cos(q3))*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3)))*(-id22*(u1*u2*cos(q2) + sin(q2)*u1d \
        + u3d + u5d)*sin(q2) - if22*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) - mc*(-l1*sin(q2) - \
        rr*sin(q2)*sin(q3))*(-l1*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        l2*(u1*sin(q2) + u3)**2 + rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3))) - \
        mc*(l2*sin(q2) - rr*sin(q2)*cos(q3))*(-l1*(u1*sin(q2) + u3)**2 + \
        l2*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*sin(q3) + rr*(u1*sin(q2) + u3 + \
        u5)*u3*sin(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*cos(q3) - (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) - (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3))) - mc*(l1*cos(q2)*cos(q3) + \
        l2*sin(q3)*cos(q2))*(l1*(u1*sin(q2) + u3)*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + l1*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) \
        + u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        l2*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - u2*u3*sin(q3) \
        - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + rr*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*(u1*sin(q2) + u3 + u5)*sin(q3) - \
        rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*u3*sin(q3) - \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*(u1*sin(q2) + u3 + \
        u5)*cos(q3) + rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*u3*cos(q3) + \
        rr*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - u2*u3*sin(q3) \
        - sin(q3)*cos(q2)*u1d + cos(q3)*u2d)*cos(q3) + \
        rr*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d)*sin(q3)) + \
        md*rr*(-rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + u5)*sin(q3) + \
        rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - rr*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d + u5d)*cos(q3) - (rr*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*cos(q3) + rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)))*sin(q2)*cos(q3) + md*rr*(rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + \
        u5)*u3*cos(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q3) + (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*sin(q2)*sin(q3) - me*(-l3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*(l4*(sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2)) - \
        rf*(sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)* \
        cos(q2)*cos(q3))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - me*(l3*cos(q2)*cos(q3) - l4*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4)))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-if11*(u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + u4)*u6 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + if22*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-ie11*((u1*sin(q2) + u3)*u4*cos(q4) + \
        (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + ie22*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        ie31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - \
        (ie31*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + ie33*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4))) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-ie22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3) + u4) + (ie31*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie33*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        (-ic22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) + (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*sin(q2) + (id11*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*(u1*sin(q2) + u3 + u5) - id11*((-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u5 - u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) - \
        id22*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*(u1*sin(q2) + u3 + \
        u5))*cos(q2)*cos(q3) - (-id11*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*(u1*sin(q2) + u3 + u5) - id11*(-(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*u5 + u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + \
        id22*(u1*cos(q2)*cos(q3) + u2*sin(q3))*(u1*sin(q2) + u3 + \
        u5))*sin(q3)*cos(q2) + (if11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*(((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))*u6 \
        - u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + \
        sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - if22*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6))*cos(q2)*cos(q3) - \
        (-ic11*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + \
        ic22*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        ic31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) - \
        (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*sin(q2) + \
        u3))*sin(q3)*cos(q2) + (-ic22*(u1*sin(q2) + \
        u3)*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) - \
        ic31*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) - \
        ic33*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*sin(q2) + \
        u3))*cos(q2)*cos(q3) + (-ie22*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) - ie31*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - ie33*(-u1*u2*sin(q2)*cos(q3) \
        - u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4)))*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*cos(q2)**2*cos(q3)**2 - \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*(rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3)))*(-ic22*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - id22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d) - if22*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d + \
        u6d)*cos(q4) - mc*(-l1 - rr*sin(q3))*(-l1*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - l2*(u1*sin(q2) + u3)**2 + rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + \
        u5)*u3*cos(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q3) + (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3))) - mc*(l2 \
        - rr*cos(q3))*(-l1*(u1*sin(q2) + u3)**2 + l2*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*sin(q3) + rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q3) - \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3))) + md*rr*(-rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*sin(q3) + rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q3) - \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3)))*cos(q3) + md*rr*(rr*(u1*sin(q2) + u3)*(u1*sin(q2) \
        + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*sin(q3) \
        - me*(-l3*cos(q4) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*(-l4*sin(q4) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*sin( \
        q4)*cos(q2)*cos(q3))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        me*(l4*cos(q4) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*cos( \
        q2)*cos(q3)*cos(q4))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q4) + mf*rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3)*cos(q4) - mf*rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + \
        cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3))*sin(q4)*cos(q2)*cos(q3) - \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) + (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + (-ie22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3) + u4) + (ie31*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie33*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q4) + (-if11*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + u4)*u6 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + if22*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4) + u6))*sin(q4) + \
        (-ie11*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + \
        ie22*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4) - ie31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - \
        (ie31*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + ie33*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + \
        u3*cos(q4)))*sin(q4))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)))

    return Fz_r_c

def contact_force_front_longitudinal_constraints(lam, mooreParameters, taskSignals):

    """Return longitudinal contact force of front wheel under the constraint condition.

    lam : float
        The tilt angle.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fx_f_c : float
        The front wheel longitudinal contact force along forward direction of
        intersection line between front wheel plane and horizontal plane under 
        the constraint condition.

    """

    mp = mooreParameters
    ts = taskSignals

    d1 = mp['d1']; d2 = mp['d2']; d3 = mp['d3']
    l1 = mp['l1']; l2 = mp['l2']; l3 = mp['l3']; l4 = mp['l4']
    rr = mp['rr']; rf = mp['rf']; g = mp['g']
    mc = mp['mc'];  md = mp['md']; me = mp['me'];  mf = mp['mf']
    ic11 = mp['ic11']; ic22 = mp['ic22']; ic33 = mp['ic33']; ic31 = mp['ic31']
    id11 = mp['id11']; id22 = mp['id22']; if11 = mp['if11']; if22 = mp['if22']
    ie11 = mp['ie11']; ie22 = mp['ie22']; ie33 = mp['ie33']; ie31 = mp['ie31']

    q2 = ts['RollAngle']; q3 = lam; q4 = ts['SteerAngle']
    u1 = ts['YawRate'];  u2 = ts['RollRate']; u3 = ts['PitchRate']
    u4 = ts['SteerRate']; u5 = ts['RearWheelRate']; u6 = ts['FrontWheelRate']
    u1d = ts['YawAcc']; u2d = ts['RollAcc']; u3d = ts['PitchAcc']
    u4d = ts['SteerAcc']; u5d = ts['RearWheelAcc']; u6d = ts['FrontWheelAcc']

    Fx_f_c = -(-me*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q3) - \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q2)*sin(q4)*cos(q3)**2)*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*cos(q3)*cos(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q3))*(-l3*((sin(q2)* \
        cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))**2 + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - mf*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q3) - \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q2)*sin(q4)*cos(q3)**2)*(rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        mf*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*cos(q3)*cos(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q3))*(rf*(sin(q2)* \
        sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) + (-((-rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) - (d2 - \
        rr*cos(q3))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3)) + (-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3)))*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*cos(q3)*cos(q4) - \
        (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q3)) - \
        ((rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q3) + \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q2)*sin(q4)*cos(q3)**2))*(-if22*(-(u1*sin(q2 \
        ) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) \
        - me*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        me*rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3) - mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*cos(q3)*cos(q4) - \
        (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q3))*(rf*((sin(q2)* \
        sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q3) + \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q2)*sin(q4)*cos(q3)**2)*(rf*((sin(q2)*sin(q4 \
        ) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*cos( \
        q2)*cos(q3))*(-id22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q2) - if22*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) - mc*(-l1*sin(q2) - \
        rr*sin(q2)*sin(q3))*(-l1*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        l2*(u1*sin(q2) + u3)**2 + rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3))) - \
        mc*(l2*sin(q2) - rr*sin(q2)*cos(q3))*(-l1*(u1*sin(q2) + u3)**2 + \
        l2*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*sin(q3) + rr*(u1*sin(q2) + u3 + \
        u5)*u3*sin(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*cos(q3) - (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) - (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3))) - mc*(l1*cos(q2)*cos(q3) + \
        l2*sin(q3)*cos(q2))*(l1*(u1*sin(q2) + u3)*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + l1*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) \
        + u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        l2*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - u2*u3*sin(q3) \
        - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + rr*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*(u1*sin(q2) + u3 + u5)*sin(q3) - \
        rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*u3*sin(q3) - \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*(u1*sin(q2) + u3 + \
        u5)*cos(q3) + rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*u3*cos(q3) + \
        rr*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - u2*u3*sin(q3) \
        - sin(q3)*cos(q2)*u1d + cos(q3)*u2d)*cos(q3) + \
        rr*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d)*sin(q3)) + \
        md*rr*(-rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + u5)*sin(q3) + \
        rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - rr*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d + u5d)*cos(q3) - (rr*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*cos(q3) + rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)))*sin(q2)*cos(q3) + md*rr*(rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + \
        u5)*u3*cos(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q3) + (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*sin(q2)*sin(q3) - me*(-l3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*(l4*(sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2)) - \
        rf*(sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)* \
        cos(q2)*cos(q3))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - me*(l3*cos(q2)*cos(q3) - l4*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4)))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-if11*(u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + u4)*u6 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + if22*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-ie11*((u1*sin(q2) + u3)*u4*cos(q4) + \
        (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + ie22*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        ie31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - \
        (ie31*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + ie33*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4))) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-ie22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3) + u4) + (ie31*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie33*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        (-ic22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) + (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*sin(q2) + (id11*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*(u1*sin(q2) + u3 + u5) - id11*((-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u5 - u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) - \
        id22*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*(u1*sin(q2) + u3 + \
        u5))*cos(q2)*cos(q3) - (-id11*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*(u1*sin(q2) + u3 + u5) - id11*(-(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*u5 + u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + \
        id22*(u1*cos(q2)*cos(q3) + u2*sin(q3))*(u1*sin(q2) + u3 + \
        u5))*sin(q3)*cos(q2) + (if11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*(((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))*u6 \
        - u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + \
        sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - if22*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6))*cos(q2)*cos(q3) - \
        (-ic11*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + \
        ic22*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        ic31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) - \
        (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*sin(q2) + \
        u3))*sin(q3)*cos(q2) + (-ic22*(u1*sin(q2) + \
        u3)*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) - \
        ic31*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) - \
        ic33*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*sin(q2) + \
        u3))*cos(q2)*cos(q3) + (-ie22*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) - ie31*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - ie33*(-u1*u2*sin(q2)*cos(q3) \
        - u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4)))*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*cos(q3)*cos(q4) - \
        (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + \
        sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q3))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)) + rf*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q3) + \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q2)*sin(q4)*cos(q3)**2)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*cos(q2)*cos(q3))*(g*mc*((-l1 - \
        rr*sin(q3))*cos(q2)*cos(q3) - (l2 - rr*cos(q3))*sin(q3)*cos(q2)) \
        + g*me*((-l3*cos(q4) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4))*cos(q2)*cos(q3) + \
        (-l4*sin(q4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 \
        + cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3))*(sin(q2 \
        )*cos(q4) + sin(q3)*sin(q4)*cos(q2)) + (l4*cos(q4) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2* \
        cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))) + g*mf*rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*sin( \
        q4)*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*cos(q3)*cos(q4) - \
        (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + \
        sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q3))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)) + rf*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q3) + \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q2)*sin(q4)*cos(q3)**2)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + d2*sin(q3)* \
        cos(q2))*cos(q4))*cos(q2)*cos(q3))*(-ic22*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - id22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d) - if22*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d + \
        u6d)*cos(q4) - mc*(-l1 - rr*sin(q3))*(-l1*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - l2*(u1*sin(q2) + u3)**2 + rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + \
        u5)*u3*cos(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q3) + (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3))) - mc*(l2 \
        - rr*cos(q3))*(-l1*(u1*sin(q2) + u3)**2 + l2*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*sin(q3) + rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q3) - \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3))) + md*rr*(-rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*sin(q3) + rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q3) - \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3)))*cos(q3) + md*rr*(rr*(u1*sin(q2) + u3)*(u1*sin(q2) \
        + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*sin(q3) \
        - me*(-l3*cos(q4) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*(-l4*sin(q4) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*sin( \
        q4)*cos(q2)*cos(q3))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        me*(l4*cos(q4) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*cos( \
        q2)*cos(q3)*cos(q4))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q4) + mf*rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3)*cos(q4) - mf*rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + \
        cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3))*sin(q4)*cos(q2)*cos(q3) - \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) + (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + (-ie22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3) + u4) + (ie31*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie33*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q4) + (-if11*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + u4)*u6 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + if22*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4) + u6))*sin(q4) + \
        (-ie11*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + \
        ie22*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4) - ie31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - \
        (ie31*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + ie33*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + \
        u3*cos(q4)))*sin(q4))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)))

    return Fx_f_c

def contact_force_front_lateral_constraints(lam, mooreParameters, taskSignals):

    """Return lateral contact force of front wheel under the constraint condition.

    lam : float
        The tilt angle.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fy_f_c : float
        The front wheel lateral contact force along lateral direction which is
        perpendicular to Fx_f_c direction and points to the right of a rider 
        under the constraint condition.

    """

    mp = mooreParameters
    ts = taskSignals

    d1 = mp['d1']; d2 = mp['d2']; d3 = mp['d3']
    l1 = mp['l1']; l2 = mp['l2']; l3 = mp['l3']; l4 = mp['l4']
    rr = mp['rr']; rf = mp['rf']; g = mp['g']
    mc = mp['mc'];  md = mp['md']; me = mp['me'];  mf = mp['mf']
    ic11 = mp['ic11']; ic22 = mp['ic22']; ic33 = mp['ic33']; ic31 = mp['ic31']
    id11 = mp['id11']; id22 = mp['id22']; if11 = mp['if11']; if22 = mp['if22']
    ie11 = mp['ie11']; ie22 = mp['ie22']; ie33 = mp['ie33']; ie31 = mp['ie31']

    q2 = ts['RollAngle']; q3 = lam; q4 = ts['SteerAngle']
    u1 = ts['YawRate'];  u2 = ts['RollRate']; u3 = ts['PitchRate']
    u4 = ts['SteerRate']; u5 = ts['RearWheelRate']; u6 = ts['FrontWheelRate']
    u1d = ts['YawAcc']; u2d = ts['RollAcc']; u3d = ts['PitchAcc']
    u4d = ts['SteerAcc']; u5d = ts['RearWheelAcc']; u6d = ts['FrontWheelAcc']

    Fy_f_c = -(-me*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2 + sin(q4)**2*cos(q3)**2)**(-0.5) + \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q4)**2*cos(q3)**2)*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        me*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*(sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5) - ((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2 + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos( \
        q3)**2*cos(q4))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - me*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q2)*cos(q3) - \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q3)*sin(q4)*cos(q3))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        mf*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2 + sin(q4)**2*cos(q3)**2)**(-0.5) + \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q4)**2*cos(q3)**2)*(-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        mf*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*(sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5) - ((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2 + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos( \
        q3)**2*cos(q4))*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - mf*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q2)*cos(q3) - \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q3)*sin(q4)*cos(q3))*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        (-(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2 + sin(q4)**2*cos(q3)**2)**(-0.5) - \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q4)**2*cos(q3)**2)*(-(rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3)) - (-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + rr*sin(q3))) - \
        ((-rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) - (d2 - \
        rr*cos(q3))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3)) + (-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3)))*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*(sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5) + ((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2 + \
        sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q3)**2*cos(q4)) - \
        ((rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q2)*cos(q3) + \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q3)*sin(q4)*cos(q3)))*(-if22*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) - me*rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        me*rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3) - mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + (g*mc*((-l1 - \
        rr*sin(q3))*cos(q2)*cos(q3) - (l2 - rr*cos(q3))*sin(q3)*cos(q2)) \
        + g*me*((-l3*cos(q4) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4))*cos(q2)*cos(q3) + \
        (-l4*sin(q4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 \
        + cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3))*(sin(q2 \
        )*cos(q4) + sin(q3)*sin(q4)*cos(q2)) + (l4*cos(q4) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2* \
        cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))) + g*mf*rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*sin( \
        q4)*cos(q2)*cos(q3))*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*(sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5) + ((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2 + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos( \
        q3)**2*cos(q4))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)) + rf*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q2)*cos(q3) + \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q3)*sin(q4)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*cos(q2)*cos(q3) - \
        (-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2 + sin(q4)**2*cos(q3)**2)**(-0.5) - \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q4)**2*cos(q3)**2)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3)))/(-rf*((rf*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*(sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5) + ((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2 + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos( \
        q3)**2*cos(q4))*(rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q2)*cos(q3) + \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q3)*sin(q4)*cos(q3))*(rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - \
        (-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2 + sin(q4)**2*cos(q3)**2)**(-0.5) - \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q4)**2*cos(q3)**2)*(-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) - (d2 - \
        rr*cos(q3))*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3)))*(-id22*(u1*u2*cos(q2) + sin(q2)*u1d \
        + u3d + u5d)*sin(q2) - if22*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) - mc*(-l1*sin(q2) - \
        rr*sin(q2)*sin(q3))*(-l1*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        l2*(u1*sin(q2) + u3)**2 + rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3))) - \
        mc*(l2*sin(q2) - rr*sin(q2)*cos(q3))*(-l1*(u1*sin(q2) + u3)**2 + \
        l2*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*sin(q3) + rr*(u1*sin(q2) + u3 + \
        u5)*u3*sin(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*cos(q3) - (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) - (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3))) - mc*(l1*cos(q2)*cos(q3) + \
        l2*sin(q3)*cos(q2))*(l1*(u1*sin(q2) + u3)*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + l1*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) \
        + u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        l2*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - u2*u3*sin(q3) \
        - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + rr*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*(u1*sin(q2) + u3 + u5)*sin(q3) - \
        rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*u3*sin(q3) - \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*(u1*sin(q2) + u3 + \
        u5)*cos(q3) + rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*u3*cos(q3) + \
        rr*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - u2*u3*sin(q3) \
        - sin(q3)*cos(q2)*u1d + cos(q3)*u2d)*cos(q3) + \
        rr*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d)*sin(q3)) + \
        md*rr*(-rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + u5)*sin(q3) + \
        rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - rr*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d + u5d)*cos(q3) - (rr*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*cos(q3) + rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)))*sin(q2)*cos(q3) + md*rr*(rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + \
        u5)*u3*cos(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q3) + (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*sin(q2)*sin(q3) - me*(-l3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*(l4*(sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2)) - \
        rf*(sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)* \
        cos(q2)*cos(q3))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - me*(l3*cos(q2)*cos(q3) - l4*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4)))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-if11*(u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + u4)*u6 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + if22*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-ie11*((u1*sin(q2) + u3)*u4*cos(q4) + \
        (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + ie22*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        ie31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - \
        (ie31*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + ie33*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4))) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-ie22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3) + u4) + (ie31*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie33*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        (-ic22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) + (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*sin(q2) + (id11*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*(u1*sin(q2) + u3 + u5) - id11*((-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u5 - u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) - \
        id22*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*(u1*sin(q2) + u3 + \
        u5))*cos(q2)*cos(q3) - (-id11*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*(u1*sin(q2) + u3 + u5) - id11*(-(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*u5 + u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + \
        id22*(u1*cos(q2)*cos(q3) + u2*sin(q3))*(u1*sin(q2) + u3 + \
        u5))*sin(q3)*cos(q2) + (if11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*(((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))*u6 \
        - u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + \
        sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - if22*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6))*cos(q2)*cos(q3) - \
        (-ic11*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + \
        ic22*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        ic31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) - \
        (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*sin(q2) + \
        u3))*sin(q3)*cos(q2) + (-ic22*(u1*sin(q2) + \
        u3)*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) - \
        ic31*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) - \
        ic33*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*sin(q2) + \
        u3))*cos(q2)*cos(q3) + (-ie22*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) - ie31*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - ie33*(-u1*u2*sin(q2)*cos(q3) \
        - u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4)))*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*(sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5) + ((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2 + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos( \
        q3)**2*cos(q4))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)) + rf*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 \
        + sin(q4)**2*cos(q3)**2)**(-0.5)*sin(q2)*cos(q3) + \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q3)*sin(q4)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*cos(q2)*cos(q3) - \
        (-(-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2*((-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))**2 + sin(q4)**2*cos(q3)**2)**(-0.5) - \
        ((-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))**2 + sin(q4)**2*cos \
        (q3)**2)**(-0.5)*sin(q4)**2*cos(q3)**2)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3)))*(-ic22*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - id22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d) - if22*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d + \
        u6d)*cos(q4) - mc*(-l1 - rr*sin(q3))*(-l1*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - l2*(u1*sin(q2) + u3)**2 + rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + \
        u5)*u3*cos(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q3) + (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3))) - mc*(l2 \
        - rr*cos(q3))*(-l1*(u1*sin(q2) + u3)**2 + l2*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*sin(q3) + rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q3) - \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3))) + md*rr*(-rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*sin(q3) + rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q3) - \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3)))*cos(q3) + md*rr*(rr*(u1*sin(q2) + u3)*(u1*sin(q2) \
        + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*sin(q3) \
        - me*(-l3*cos(q4) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*(-l4*sin(q4) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*sin( \
        q4)*cos(q2)*cos(q3))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        me*(l4*cos(q4) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*cos( \
        q2)*cos(q3)*cos(q4))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q4) + mf*rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3)*cos(q4) - mf*rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + \
        cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3))*sin(q4)*cos(q2)*cos(q3) - \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) + (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + (-ie22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3) + u4) + (ie31*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie33*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q4) + (-if11*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + u4)*u6 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + if22*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4) + u6))*sin(q4) + \
        (-ie11*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + \
        ie22*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4) - ie31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - \
        (ie31*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + ie33*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + \
        u3*cos(q4)))*sin(q4))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)))

    return Fy_f_c


def contact_force_front_normal_constraints(lam, mooreParameters, taskSignals):

    """Return normal contact force of front wheel under the constraint condition.

    lam : float
        The tilt angle.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fz_f_c : float
        The front wheel normal contact force along vertical direction A['3'], 
        perpendicular to the horizontal plane and pointing to downward, 
        under the constraint condition.

    """

    mp = mooreParameters
    ts = taskSignals

    d1 = mp['d1']; d2 = mp['d2']; d3 = mp['d3']
    l1 = mp['l1']; l2 = mp['l2']; l3 = mp['l3']; l4 = mp['l4']
    rr = mp['rr']; rf = mp['rf']; g = mp['g']
    mc = mp['mc'];  md = mp['md']; me = mp['me'];  mf = mp['mf']
    ic11 = mp['ic11']; ic22 = mp['ic22']; ic33 = mp['ic33']; ic31 = mp['ic31']
    id11 = mp['id11']; id22 = mp['id22']; if11 = mp['if11']; if22 = mp['if22']
    ie11 = mp['ie11']; ie22 = mp['ie22']; ie33 = mp['ie33']; ie31 = mp['ie31']

    q2 = ts['RollAngle']; q3 = lam; q4 = ts['SteerAngle']
    u1 = ts['YawRate'];  u2 = ts['RollRate']; u3 = ts['PitchRate']
    u4 = ts['SteerRate']; u5 = ts['RearWheelRate']; u6 = ts['FrontWheelRate']
    u1d = ts['YawAcc']; u2d = ts['RollAcc']; u3d = ts['PitchAcc']
    u4d = ts['SteerAcc']; u5d = ts['RearWheelAcc']; u6d = ts['FrontWheelAcc']

    Fz_f_c = -(g*me + g*mf - me*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - me*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        me*(-l3*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - \
        l4*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4))**2 + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q2)*cos(q3) - mf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - mf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        mf*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q2)*cos(q3) + (-((-rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) - (d2 - \
        rr*cos(q3))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3)) + (-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3)))*(-sin(q2)*sin(q4) + sin(q3)*cos(q2)*cos(q4)) + \
        ((rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*cos(q2)*cos(q3) - \
        (-(rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3)) - (-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3)))*(-sin(q2)*cos(q4) - \
        sin(q3)*sin(q4)*cos(q2)))*(-if22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) - me*rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        me*rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3) - mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + (g*mc*((-l1 - \
        rr*sin(q3))*cos(q2)*cos(q3) - (l2 - rr*cos(q3))*sin(q3)*cos(q2)) \
        + g*me*((-l3*cos(q4) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4))*cos(q2)*cos(q3) + \
        (-l4*sin(q4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 \
        + cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3))*(sin(q2 \
        )*cos(q4) + sin(q3)*sin(q4)*cos(q2)) + (l4*cos(q4) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2* \
        cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))) + g*mf*rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*sin( \
        q4)*cos(q2)*cos(q3))*(rf*(-sin(q2)*sin(q4) + \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*cos(q2)**2*cos(q3)**2 - \
        (-sin(q2)*cos(q4) - sin(q3)*sin(q4)*cos(q2))*(rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3)))/(-rf*((rf*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (-rf*(-sin(q2)*sin(q4) + \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rf*(rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)**2*cos(q3)**2 - \
        (-sin(q2)*cos(q4) - \
        sin(q3)*sin(q4)*cos(q2))*(-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) - (d2 - \
        rr*cos(q3))*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3)))*(-id22*(u1*u2*cos(q2) + sin(q2)*u1d \
        + u3d + u5d)*sin(q2) - if22*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) - mc*(-l1*sin(q2) - \
        rr*sin(q2)*sin(q3))*(-l1*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        l2*(u1*sin(q2) + u3)**2 + rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3))) - \
        mc*(l2*sin(q2) - rr*sin(q2)*cos(q3))*(-l1*(u1*sin(q2) + u3)**2 + \
        l2*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*sin(q3) + rr*(u1*sin(q2) + u3 + \
        u5)*u3*sin(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*cos(q3) - (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) - (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3))) - mc*(l1*cos(q2)*cos(q3) + \
        l2*sin(q3)*cos(q2))*(l1*(u1*sin(q2) + u3)*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + l1*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) \
        + u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        l2*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - u2*u3*sin(q3) \
        - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + rr*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*(u1*sin(q2) + u3 + u5)*sin(q3) - \
        rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*u3*sin(q3) - \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*(u1*sin(q2) + u3 + \
        u5)*cos(q3) + rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*u3*cos(q3) + \
        rr*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - u2*u3*sin(q3) \
        - sin(q3)*cos(q2)*u1d + cos(q3)*u2d)*cos(q3) + \
        rr*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d)*sin(q3)) + \
        md*rr*(-rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + u5)*sin(q3) + \
        rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - rr*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d + u5d)*cos(q3) - (rr*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*cos(q3) + rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)))*sin(q2)*cos(q3) + md*rr*(rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + \
        u5)*u3*cos(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q3) + (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*sin(q2)*sin(q3) - me*(-l3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*(l4*(sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2)) - \
        rf*(sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)* \
        cos(q2)*cos(q3))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - me*(l3*cos(q2)*cos(q3) - l4*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4)))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-if11*(u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + u4)*u6 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + if22*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-ie11*((u1*sin(q2) + u3)*u4*cos(q4) + \
        (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + ie22*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        ie31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - \
        (ie31*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + ie33*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4))) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*(-ie22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3) + u4) + (ie31*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie33*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        (-ic22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) - \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) + (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*sin(q2) + (id11*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*(u1*sin(q2) + u3 + u5) - id11*((-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u5 - u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) - \
        id22*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*(u1*sin(q2) + u3 + \
        u5))*cos(q2)*cos(q3) - (-id11*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*(u1*sin(q2) + u3 + u5) - id11*(-(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*u5 + u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + \
        id22*(u1*cos(q2)*cos(q3) + u2*sin(q3))*(u1*sin(q2) + u3 + \
        u5))*sin(q3)*cos(q2) + (if11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*(((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))*u6 \
        - u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + \
        sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - if22*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6))*cos(q2)*cos(q3) - \
        (-ic11*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) + \
        ic22*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        ic31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) - \
        (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*sin(q2) + \
        u3))*sin(q3)*cos(q2) + (-ic22*(u1*sin(q2) + \
        u3)*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) - \
        ic31*(u1*u2*sin(q2)*sin(q3) - u1*u3*cos(q2)*cos(q3) - \
        u2*u3*sin(q3) - sin(q3)*cos(q2)*u1d + cos(q3)*u2d) - \
        ic33*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*sin(q2) + \
        u3))*cos(q2)*cos(q3) + (-ie22*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) - ie31*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - ie33*(-u1*u2*sin(q2)*cos(q3) \
        - u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4)))*cos(q2)*cos(q3))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)) + \
        (rf*(-sin(q2)*sin(q4) + sin(q3)*cos(q2)*cos(q4))*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*cos(q2)**2*cos(q3)**2 - \
        (-sin(q2)*cos(q4) - sin(q3)*sin(q4)*cos(q2))*(rf*(sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3)))*(-ic22*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - id22*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d) - if22*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d + \
        u6d)*cos(q4) - mc*(-l1 - rr*sin(q3))*(-l1*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - l2*(u1*sin(q2) + u3)**2 + rr*(u1*sin(q2) + \
        u3)*(u1*sin(q2) + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + \
        u5)*u3*cos(q3) - rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + \
        u5d)*sin(q3) + (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        l2*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3))) - mc*(l2 \
        - rr*cos(q3))*(-l1*(u1*sin(q2) + u3)**2 + l2*(u1*u2*cos(q2) + \
        sin(q2)*u1d + u3d) - rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*sin(q3) + rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q3) - \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - l2*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)))*(u1*cos(q2)*cos(q3) + u2*sin(q3)) - \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3))) + md*rr*(-rr*(u1*sin(q2) + u3)*(u1*sin(q2) + u3 + \
        u5)*sin(q3) + rr*(u1*sin(q2) + u3 + u5)*u3*sin(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q3) - \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + u2*sin(q3))*sin(q3))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3)))*cos(q3) + md*rr*(rr*(u1*sin(q2) + u3)*(u1*sin(q2) \
        + u3 + u5)*cos(q3) - rr*(u1*sin(q2) + u3 + u5)*u3*cos(q3) - \
        rr*(u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q3) + \
        (rr*(-u1*sin(q3)*cos(q2) + u2*cos(q3))*cos(q3) + \
        rr*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3))*sin(q3))*(-u1*sin(q3)*cos(q2) + u2*cos(q3)))*sin(q3) \
        - me*(-l3*cos(q4) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4))*(-l3*(-(u1*sin(q2) + \
        u3)*u4*sin(q4) + (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        me*(-l4*sin(q4) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*sin( \
        q4)*cos(q2)*cos(q3))*(l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) + l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) - \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        me*(l4*cos(q4) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5)*cos( \
        q2)*cos(q3)*cos(q4))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*cos(q4) + (-sin(q2)*sin(q3)*sin(q4) + \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) - mf*rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q4) + mf*rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4))*cos(q2)*cos(q3)*cos(q4) - mf*rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-0.5*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(2.0*u2*sin(q2)*sin(q3)*cos(q4) + \
        2.0*u2*sin(q4)*cos(q2) - 2.0*u3*cos(q2)*cos(q3)*cos(q4) + \
        2.0*u4*sin(q2)*cos(q4) + 2.0*u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + \
        cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3))*sin(q4)*cos(q2)*cos(q3) - \
        (ic11*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic31*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3)) + (ic31*(-u1*sin(q3)*cos(q2) + u2*cos(q3)) + \
        ic33*(u1*cos(q2)*cos(q3) + u2*sin(q3)))*(-u1*sin(q3)*cos(q2) + \
        u2*cos(q3)) + (-ie22*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d - \
        (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*cos(q4) + \
        (-sin(q2)*sin(q3)*sin(q4) + cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - (ie11*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie31*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*(u1*cos(q2)*cos(q3) \
        + u2*sin(q3) + u4) + (ie31*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)) + \
        ie33*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*cos(q4) + (-if11*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - if11*((u1*sin(q2) + \
        u3)*u4*cos(q4) + (sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1d \
        - (-u1*sin(q3)*cos(q2) + u2*cos(q3))*u4*sin(q4) + \
        (sin(q2)*sin(q3)*cos(q4) + sin(q4)*cos(q2))*u1*u2 - \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + u4)*u6 - \
        u1*u3*cos(q2)*cos(q3)*cos(q4) - u2*u3*sin(q3)*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + if22*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + u3*cos(q4) + u6))*sin(q4) + \
        (-ie11*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d - (-u1*sin(q3)*cos(q2) + \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + \
        ie22*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4) - ie31*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d + u4d) - \
        (ie31*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) + ie33*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 \
        - u2*sin(q4)*cos(q3) + \
        u3*cos(q4)))*sin(q4))/(-rf*((rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) - (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4)) + (rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3)*cos(q4) + (d2 - \
        rr*cos(q3))*cos(q4))*(-d3*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*sin(q4) - (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4)))*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) - \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d3*cos(q2)*cos(q3) - (d2*sin(q2) \
        - rr*sin(q2)*cos(q3))*sin(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*cos(q4))*(d1 + d3*cos(q4) + \
        rf*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q4) + \
        rr*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + (d2*sin(q2) - \
        rr*sin(q2)*cos(q3))*cos(q4) + (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*(d1*sin(q2) + d3*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*cos(q2)*cos(q3) - rf*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(d1*sin(q2) + d3*(sin(q2)*cos(q4) \
        + sin(q3)*sin(q4)*cos(q2)) + rf*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)**(-0.5) + \
        rr*sin(q2)*sin(q3))*(-rf*(sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*cos(q2)*cos(q3) + \
        rf*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*sin(q4)*cos(q2)*cos(q3) + (d2 - \
        rr*cos(q3))*sin(q4) - (d2*sin(q2) - rr*sin(q2)*cos(q3))*cos(q4) - \
        (d1*cos(q2)*cos(q3) + \
        d2*sin(q3)*cos(q2))*sin(q4))*cos(q2)*cos(q3)))

    return Fz_f_c
