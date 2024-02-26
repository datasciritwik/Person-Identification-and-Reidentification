import openvino.runtime as openrun
import cv2
import numpy as np

def load_model(xml_path, device='CPU'):
    core = openrun.Core()
    model = core.read_model(xml_path)
    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer = model.input(0)
    _,_,height,width = input_layer.shape
    return model, compiled_model, (width, height), input_layer

def preprocessing(image, target_shape):
    resized_image = cv2.resize(image, target_shape)
    resized_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_BGR2RGB)
    resized_image = resized_image.transpose((2, 0, 1))
    resized_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
    return resized_image

def cosine_similarity(v1, v2):
    """
    Calculate cosine similarity between two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

def find_most_similar_pair(target_vector, vectors):
    """
    Find the most similar pair to the target vector from a list of vectors.
    """
    max_similarity = -1
    curr_objID = 0.0
    for _, vector in enumerate(vectors):
        # print(vector)
        curr_vector = vector[0]
        curr_objID = vector[1]
        similarity = cosine_similarity(target_vector, curr_vector.flatten())
        if similarity > max_similarity:
            max_similarity = similarity
    return curr_objID, max_similarity