import numpy as np 

if __name__ == "__main__":
    pass

def XYZ_to_CIELUV(c, whitePoint):
    L_star, u_star, v_star = 0, 0, 0
    X, Y, Z = c[0], c[1], c[2]
    X_n, Y_n, Z_n = whitePoint[0], whitePoint[1], whitePoint[2]
    
    u_stick_n = 4 * X_n / (X_n + 15 * Y_n + 3 * Z_n)
    v_stick_n = 9 * X_n / (X_n + 15 * Y_n + 3 * Z_n)
    
    if Y / Y_n <= (6 / 29) ** 3:
        L_star = (29 / 3) ** 3 * Y / Y_n
    else:
        L_star = 116 * (Y / Y_n) ** (1 / 3) - 16
    
    u_stick = 4 * X / (X + 15 * Y + 3 * Z)
    v_stick = 9 * Y / (X + 15 * Y + 3 * Z)
    
    u_star = 13 * L_star * (u_stick - u_stick_n)
    v_star = 13 * L_star * (v_stick - v_stick_n)
    
    return np.array([L_star, u_star, v_star])

def CIELUV_DeltaE_from_XYZ(XYZ1, XYZ2, whitePoint):
    return np.linalg.norm(XYZ_to_CIELUV(XYZ1, whitePoint) - XYZ_to_CIELUV(XYZ2, whitePoint))