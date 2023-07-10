import math
import numpy as np

def Angle(A,B,C):
    BA = np.array([A[0]-B[0], A[1]-B[1], A[2]-B[2]])
    modBA= math.sqrt(BA[0]**2+BA[1]**2+BA[2]**2)
    BC = np.array([C[0]-B[0], C[1]-B[1], C[2]-B[2]])
    modBC= math.sqrt(BC[0]**2+BC[1]**2+BC[2]**2)
    BABC= modBA * modBC
    dotProduct=BA[0]*BC[0] + BA[1]*BC[1]
    x = dotProduct / BABC
    angle = math.acos(x)
    return(math.degrees(angle))

def calc_localAngle(A, S):
    # print(A, S)
    localAngle = []
    for i in range(len(S)-1):
        angle = Angle(S[i], A, S[i+1])
        if(angle <= 180):
            localAngle.append(math.radians(angle))
    angle = Angle(S[len(S)-1], A, S[0])
    if(angle <= 180):
        localAngle.append(math.radians(angle))
    # print("localAngle", localAngle)
    return(localAngle)

