import cv2
import mediapipe as mp
from scipy import sparse
import torch

def mediapipe_facemesh(image):
    image = cv2.resize(image, (600, 600))

    mp_face_mesh = mp.solutions.face_mesh
    connection_tesselation = mp_face_mesh.FACEMESH_TESSELATION

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    x_axis = []
    y_axis = []
    z_axis = []

    FEATURE_MATRIX = torch.zeros((468, 468))

    for i, nodes in enumerate(results.multi_face_landmarks[0].landmark):
        x_axis.append(nodes.x)
        y_axis.append(nodes.y)
        z_axis.append(nodes.z)

    WEIGHTED_ADJACENCY_MATRIX = torch.zeros((468, 468))

    for edge in connection_tesselation:
        x1 = x_axis[edge[0]]
        y1 = y_axis[edge[0]]
        z1 = z_axis[edge[0]]
        x2 = x_axis[edge[1]]
        y2 = y_axis[edge[1]]
        z2 = z_axis[edge[1]]

        eucleadian_distance = (((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5)

        WEIGHTED_ADJACENCY_MATRIX[edge[0]][edge[1]] = eucleadian_distance
        WEIGHTED_ADJACENCY_MATRIX[edge[1]][edge[0]] = eucleadian_distance

    return sparse.csr_matrix(WEIGHTED_ADJACENCY_MATRIX), FEATURE_MATRIX
