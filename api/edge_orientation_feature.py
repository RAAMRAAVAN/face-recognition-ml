import math

def get_edge_orientation(edge):
    reference_axis = (1, 0, 0)
    # Calculate the direction vector of the edge
    direction_vector = [edge[1][0] - edge[0][0], edge[1][1] - edge[0][1], edge[1][2] - edge[0][2]]
    
    # Calculate the dot product between the direction vector and the reference axis
    dot_product = direction_vector[0] * reference_axis[0] + direction_vector[1] * reference_axis[1] + direction_vector[2] * reference_axis[2]
    
    # Calculate the magnitude of the direction vector
    direction_magnitude = math.sqrt(direction_vector[0]**2 + direction_vector[1]**2 + direction_vector[2]**2)
    
    # Calculate the angle between the direction vector and the reference axis
    angle = math.acos(dot_product / (direction_magnitude * math.sqrt(reference_axis[0]**2 + reference_axis[1]**2 + reference_axis[2]**2)))
    
    # Return the angle in degrees
    return math.degrees(angle)

# edge = ((0, 0, 0), (1, 1, 1))
# angle = get_edge_orientation(edge)
# print(angle)
def calc_edge_orientation(A, S):
    # print(A, S)
    edge_orientation = []
    temp = []
    for i in range(len(S)):
        temp.append(S[i])
        temp.append(A)
        edge_orientation.append(get_edge_orientation(temp))
    return(edge_orientation)