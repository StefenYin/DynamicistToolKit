from numpy import sin, cos, tan, arctan, pi, sqrt
import numpy as np


def contact_force_rear_longitudinal_N1_constraints(lam, mooreParameters, taskSignals):

    """Return longitudinal contact force of rear wheel under the constraint condition.

    Note
    ----
    The contact force direction in this case is expressed by inertial frame, N['1'].

    lam : float
        The tilt angle.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fx_r_ns : float
        The rear wheel longitudinal contact force along N['1'] direction 
        under the constraint condition.

    """

    mp = mooreParameters
    ts = taskSignals

    rR = mp['rr']; l1 = mp['l1']; l2 = mp['l2'];  mc = mp['mc'];  md = mp['md']

    q1 = ts['YawAngle']; q2 = ts['RollAngle']; q3 = lam

    u1 = ts['YawRate'];  u2 = ts['RollRate']; u3 = ts['PitchRate']
    u5 = ts['RearWheelRate']

    u1d = ts['YawAcc']; u2d = ts['RollAcc']; u3d = ts['PitchAcc']
    u5d = ts['RearWheelAcc']

    Fx_r_ns_N1 = mc*rR*((u1*sin(q2) + u3 + u5)*u1*sin(q2) + u2**2)*sin(q1)*sin(q2) + \
        mc*rR*((u1*sin(q2) + u3 + u5)*u1*cos(q2) - u2d)*sin(q1)*cos(q2) - \
        mc*rR*(2.0*u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*cos(q1) + \
        mc*(sin(q1)*sin(q2)*sin(q3) - cos(q1)*cos(q3))*(l1*(u1*sin(q2) + \
        u3)**2 - l2*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) + \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) + l2*(u1*sin(q3)*cos(q2) - \
        u2*cos(q3)))*(u1*cos(q2)*cos(q3) + u2*sin(q3))) - \
        mc*(sin(q1)*sin(q2)*cos(q3) + sin(q3)*cos(q1))*(l1*(u1*u2*cos(q2) \
        + sin(q2)*u1d + u3d) + l2*(u1*sin(q2) + u3)**2 + \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) + l2*(u1*sin(q3)*cos(q2) - \
        u2*cos(q3)))*(u1*sin(q3)*cos(q2) - u2*cos(q3))) - \
        mc*(-l1*(u1*sin(q2) + u3)*(u1*sin(q3)*cos(q2) - u2*cos(q3)) + \
        l1*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        l2*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) + \
        l2*(-u1*u2*sin(q2)*sin(q3) + u1*u3*cos(q2)*cos(q3) + \
        u2*u3*sin(q3) + sin(q3)*cos(q2)*u1d - \
        cos(q3)*u2d))*sin(q1)*cos(q2) + md*rR*((u1*sin(q2) + u3 + \
        u5)*u1*sin(q2) + u2**2)*sin(q1)*sin(q2) + md*rR*((u1*sin(q2) + u3 \
        + u5)*u1*cos(q2) - u2d)*sin(q1)*cos(q2) - md*rR*(2.0*u1*u2*cos(q2) +\
        sin(q2)*u1d + u3d + u5d)*cos(q1)

    return Fx_r_ns_N1

def contact_force_rear_lateral_N2_constraints(lam, mooreParameters, taskSignals):

    """Return lateral contact force of rear wheel under the constraint condition.

    Note
    ----
    The contact force direction in this case is expressed by inertial frame, N['2'].

    lam : float
        The tilt angle.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fy_r_ns : float
        The rear wheel lateral contact force along N['2'] direction 
        under the constraint condition.

    """

    mp = mooreParameters
    ts = taskSignals

    rR = mp['rr']; l1 = mp['l1']; l2 = mp['l2'];  mc = mp['mc'];  md = mp['md']

    q1 = ts['YawAngle']; q2 = ts['RollAngle']; q3 = lam

    u1 = ts['YawRate'];  u2 = ts['RollRate']; u3 = ts['PitchRate']
    u5 = ts['RearWheelRate']

    u1d = ts['YawAcc']; u2d = ts['RollAcc']; u3d = ts['PitchAcc']
    u5d = ts['RearWheelAcc']

    Fy_r_ns_N2 = -mc*rR*((u1*sin(q2) + u3 + u5)*u1*sin(q2) + u2**2)*sin(q2)*cos(q1) - \
        mc*rR*((u1*sin(q2) + u3 + u5)*u1*cos(q2) - u2d)*cos(q1)*cos(q2) - \
        mc*rR*(2.0*u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q1) - \
        mc*(sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*(l1*(u1*u2*cos(q2) \
        + sin(q2)*u1d + u3d) + l2*(u1*sin(q2) + u3)**2 + \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) + l2*(u1*sin(q3)*cos(q2) - \
        u2*cos(q3)))*(u1*sin(q3)*cos(q2) - u2*cos(q3))) - \
        mc*(sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*(l1*(u1*sin(q2) + \
        u3)**2 - l2*(u1*u2*cos(q2) + sin(q2)*u1d + u3d) + \
        (l1*(u1*cos(q2)*cos(q3) + u2*sin(q3)) + l2*(u1*sin(q3)*cos(q2) - \
        u2*cos(q3)))*(u1*cos(q2)*cos(q3) + u2*sin(q3))) + \
        mc*(-l1*(u1*sin(q2) + u3)*(u1*sin(q3)*cos(q2) - u2*cos(q3)) + \
        l1*(-u1*u2*sin(q2)*cos(q3) - u1*u3*sin(q3)*cos(q2) + \
        u2*u3*cos(q3) + sin(q3)*u2d + cos(q2)*cos(q3)*u1d) + \
        l2*(u1*sin(q2) + u3)*(u1*cos(q2)*cos(q3) + u2*sin(q3)) + \
        l2*(-u1*u2*sin(q2)*sin(q3) + u1*u3*cos(q2)*cos(q3) + \
        u2*u3*sin(q3) + sin(q3)*cos(q2)*u1d - \
        cos(q3)*u2d))*cos(q1)*cos(q2) - md*rR*((u1*sin(q2) + u3 + \
        u5)*u1*sin(q2) + u2**2)*sin(q2)*cos(q1) - md*rR*((u1*sin(q2) + u3 \
        + u5)*u1*cos(q2) - u2d)*cos(q1)*cos(q2) - \
        md*rR*(2.0*u1*u2*cos(q2) + sin(q2)*u1d + u3d + u5d)*sin(q1)

    return Fy_r_ns_N2

def contact_force_front_longitudinal_N1_constraints(lam, mooreParameters, taskSignals):

    """Return longitudinal contact force of front wheel under the constraint condition.

    Note
    ----
    The contact force direction in this case is expressed by inertial frame, N['1'].

    lam : float
        The tilt angle.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fx_f_ns : float
        The front wheel longitudinal contact force along N['1'] direction 
        under the constraint condition.

    """

    mp = mooreParameters
    ts = taskSignals

    rF = mp['rf']; l3 = mp['l3']; l4 = mp['l4'];  me = mp['me'];  mf = mp['mf']

    q1 = ts['YawAngle']; q2 = ts['RollAngle']; q3 = lam; q4 = ts['SteerAngle']

    u1 = ts['YawRate'];  u2 = ts['RollRate']; u3 = ts['PitchRate']
    u4 = ts['SteerRate']; u6 = ts['FrontWheelRate']

    u1d = ts['YawAcc']; u2d = ts['RollAcc']; u3d = ts['PitchAcc']
    u4d = ts['SteerAcc']; u6d = ts['FrontWheelAcc']

    Fx_f_ns_N1 = -me*((sin(q1)*sin(q2)*sin(q3) - cos(q1)*cos(q3))*sin(q4) - \
        sin(q1)*cos(q2)*cos(q4))*(-l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) - l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) + \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u1*sin(q3)*cos(q2) - \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + \
        rF*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) + rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) - rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) - rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) - \
        me*((sin(q1)*sin(q2)*sin(q3) - cos(q1)*cos(q3))*cos(q4) + \
        sin(q1)*sin(q4)*cos(q2))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (u1*sin(q3)*cos(q2) - \
        u2*cos(q3))*u4*cos(q4) - (sin(q2)*sin(q3)*sin(q4) - \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rF*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) + \
        (rF*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) + me*(sin(q1)*sin(q2)*cos(q3) + \
        sin(q3)*cos(q1))*(-l3*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d + \
        (u1*sin(q3)*cos(q2) - u2*cos(q3))*u4*cos(q4) - \
        (sin(q2)*sin(q3)*sin(q4) - cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) - (rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) - \
        mf*rF*((sin(q1)*sin(q2)*sin(q3) - cos(q1)*cos(q3))*sin(q4) - \
        sin(q1)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)/((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2) + ((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) + ((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) + (u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) - ((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3) - \
        (-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3)/((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)) - \
        mf*((sin(q1)*sin(q2)*sin(q3) - cos(q1)*cos(q3))*cos(q4) + \
        sin(q1)*sin(q4)*cos(q2))*(rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) + \
        (rF*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) + mf*(sin(q1)*sin(q2)*cos(q3) + \
        sin(q3)*cos(q1))*(rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) - (rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)))

    return Fx_f_ns_N1

def contact_force_front_lateral_N2_constraints(lam, mooreParameters, taskSignals):

    """Return lateral contact force of front wheel under the constraint condition.

    Note
    ----
    The contact force direction in this case is expressed by inertial frame, N['2'].

    lam : float
        The tilt angle.
    mooreParameters : dictionary
        A dictionary of bicycle parameters with a rider in MOORE' set, not
        Benchmark's set.
    taskSignals : dictionary
        A dictionary of various states signals.

    Returns
    -------
    Fy_f_ns : float
        The front wheel lateral contact force along N['2'] direction 
        under the constraint condition.

    """

    mp = mooreParameters
    ts = taskSignals

    rF = mp['rf']; l3 = mp['l3']; l4 = mp['l4'];  me = mp['me'];  mf = mp['mf']

    q1 = ts['YawAngle']; q2 = ts['RollAngle']; q3 = lam; q4 = ts['SteerAngle']

    u1 = ts['YawRate'];  u2 = ts['RollRate']; u3 = ts['PitchRate']
    u4 = ts['SteerRate']; u6 = ts['FrontWheelRate']

    u1d = ts['YawAcc']; u2d = ts['RollAcc']; u3d = ts['PitchAcc']
    u4d = ts['SteerAcc']; u6d = ts['FrontWheelAcc']

    Fy_f_ns_N2 = me*((sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*sin(q4) - \
        cos(q1)*cos(q2)*cos(q4))*(-l3*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4)) - l3*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) + \
        l4*((u1*sin(q2) + u3)*u4*cos(q4) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u1*sin(q3)*cos(q2) - \
        u2*cos(q3))*u4*sin(q4) + (sin(q2)*sin(q3)*cos(q4) + \
        sin(q4)*cos(q2))*u1*u2 - u1*u3*cos(q2)*cos(q3)*cos(q4) - \
        u2*u3*sin(q3)*cos(q4) + sin(q4)*u3d + cos(q3)*cos(q4)*u2d) + \
        rF*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) + rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) + rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) - rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) - rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3)) + \
        me*((sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*cos(q4) + \
        sin(q4)*cos(q1)*cos(q2))*(-l3*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + l4*(-(u1*sin(q2) + u3)*u4*sin(q4) + (sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (u1*sin(q3)*cos(q2) - \
        u2*cos(q3))*u4*cos(q4) - (sin(q2)*sin(q3)*sin(q4) - \
        cos(q2)*cos(q4))*u1*u2 + u1*u3*sin(q4)*cos(q2)*cos(q3) + \
        u2*u3*sin(q3)*sin(q4) - sin(q4)*cos(q3)*u2d + cos(q4)*u3d) + \
        rF*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) - \
        (l3*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + u4) + \
        (rF*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) + me*(sin(q1)*sin(q3) - \
        sin(q2)*cos(q1)*cos(q3))*(-l3*(-(u1*sin(q2) + u3)*u4*sin(q4) + \
        (sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1d + \
        (u1*sin(q3)*cos(q2) - u2*cos(q3))*u4*cos(q4) - \
        (sin(q2)*sin(q3)*sin(q4) - cos(q2)*cos(q4))*u1*u2 + \
        u1*u3*sin(q4)*cos(q2)*cos(q3) + u2*u3*sin(q3)*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d) - l4*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4))**2 \
        + rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) + (l3*(u1*cos(q2)*cos(q3) + \
        u2*sin(q3) + u4) - l4*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4)))*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*cos(q3)*cos(q4) + u3*sin(q4)) - (rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4))) + \
        mf*rF*((sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*sin(q4) - \
        cos(q1)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-u1*u2*sin(q2)*cos(q3) - \
        u1*u3*sin(q3)*cos(q2) + u2*u3*cos(q3) + sin(q3)*u2d + \
        cos(q2)*cos(q3)*u1d + u4d) + (sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)/((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2) + ((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u2*sin(q2)*cos(q3) + ((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*u3*sin(q3)*cos(q2) + (u1*cos(q2)*cos(q3) + u2*sin(q3) \
        + u4)*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        (u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) - ((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1d + (u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*u3*sin(q3)*cos(q4) - u2*u4*sin(q4)*cos(q3) + u3*u4*cos(q4) + \
        sin(q4)*u3d + cos(q3)*cos(q4)*u2d)*cos(q2)*cos(q3) - \
        (-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3)/((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + cos(q2)**2*cos(q3)**2)) + \
        mf*((sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*cos(q4) + \
        sin(q4)*cos(q1)*cos(q2))*(rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6) - rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*cos(q2)*cos(q3) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u2*sin(q2)*cos(q3) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*u3*sin(q3)*cos(q2) - rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d)*cos(q2)*cos(q3) + \
        (rF*(sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) \
        - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4)) + mf*(sin(q1)*sin(q3) - \
        sin(q2)*cos(q1)*cos(q3))*(rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-1.5)*(-(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*(u2*sin(q2)*sin(q3)*cos(q4) + \
        u2*sin(q4)*cos(q2) - u3*cos(q2)*cos(q3)*cos(q4) + \
        u4*sin(q2)*cos(q4) + u4*sin(q3)*sin(q4)*cos(q2)) + \
        u2*sin(q2)*cos(q2)*cos(q3)**2 + \
        u3*sin(q3)*cos(q2)**2*cos(q3))*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6) + rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1d + (-u2*sin(q2)*sin(q3)*sin(q4) + \
        u2*cos(q2)*cos(q4) + u3*sin(q4)*cos(q2)*cos(q3) - \
        u4*sin(q2)*sin(q4) + u4*sin(q3)*cos(q2)*cos(q4))*u1 + \
        u2*u3*sin(q3)*sin(q4) - u2*u4*cos(q3)*cos(q4) - u3*u4*sin(q4) - \
        sin(q4)*cos(q3)*u2d + cos(q4)*u3d + u6d) + rF*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + \
        u3*cos(q4))*((sin(q2)*cos(q4) + sin(q3)*sin(q4)*cos(q2))*u1 - \
        u2*sin(q4)*cos(q3) + u3*cos(q4) + u6)*cos(q2)*cos(q3) + \
        rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*cos(q4) + \
        sin(q3)*sin(q4)*cos(q2))*u1 - u2*sin(q4)*cos(q3) + u3*cos(q4) + \
        u6)*(u2*sin(q2)*sin(q3)*cos(q4) + u2*sin(q4)*cos(q2) - \
        u3*cos(q2)*cos(q3)*cos(q4) + u4*sin(q2)*cos(q4) + \
        u4*sin(q3)*sin(q4)*cos(q2)) - (rF*(sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*(u1*cos(q2)*cos(q3) + u2*sin(q3) + \
        u4) - rF*((sin(q2)*sin(q4) - sin(q3)*cos(q2)*cos(q4))**2 + \
        cos(q2)**2*cos(q3)**2)**(-0.5)*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + \
        u3*sin(q4))*cos(q2)*cos(q3))*((sin(q2)*sin(q4) - \
        sin(q3)*cos(q2)*cos(q4))*u1 + u2*cos(q3)*cos(q4) + u3*sin(q4)))

    return Fy_f_ns_N2

def contact_force_nonslip(lam, mooreParameters, taskSignals):

    f0 = np.vectorize(contact_force_rear_longitudinal_N1_constraints)
    Fx_r_ns_N1 = f0(lam, mooreParameters, taskSignals)

    f1 = np.vectorize(contact_force_rear_lateral_N2_constraints)
    Fy_r_ns_N2 = f1(lam, mooreParameters, taskSignals)

    f2 = np.vectorize(contact_force_front_longitudinal_N1_constraints)
    Fx_f_ns_N1 = f2(lam, mooreParameters, taskSignals)

    f3 = np.vectorize(contact_force_front_lateral_N2_constraints)
    Fy_f_ns_N2 = f3(lam, mooreParameters, taskSignals)

    yawAngle = taskSignals['YawAngle']
    frontWheelYawAngle = taskSignals['FrontWheelYawAngle']

    Fx_r_ns = cos(yawAngle) * Fx_r_ns_N1 + sin(yawAngle) * Fy_r_ns_N2
    Fy_r_ns = -sin(yawAngle) * Fx_r_ns_N1 + cos(yawAngle) * Fy_r_ns_N2 
    Fx_f_ns = cos(frontWheelYawAngle) * Fx_f_ns_N1 + sin(frontWheelYawAngle) * Fy_f_ns_N2
    Fy_f_ns = -sin(frontWheelYawAngle) * Fx_f_ns_N1 + cos(frontWheelYawAngle) * Fy_f_ns_N2

    return Fx_r_ns, Fy_r_ns, Fx_f_ns, Fy_f_ns
