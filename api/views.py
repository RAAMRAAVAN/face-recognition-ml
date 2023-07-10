from PIL import Image
from django.http import JsonResponse
from .gae_main import gae_for

from rest_framework.request import Request
from rest_framework.decorators import api_view
from rest_framework.response import Response

# latent, adj, features, image = gae_for("https://mymodernmet.com/wp/wp-content/uploads/2019/09/100k-ai-faces-6.jpg")
@api_view(['POST'])
def latent_api(request: Request):
    if request.method == 'POST':
        # Assuming the matrix is passed as a JSON object in the request body
        path = request.data.get('image_link')
        latent, adj, features, image = gae_for(path)
        adj = adj.toarray()
        features = features.numpy()
        latent = latent.tolist()
        adj = adj.tolist()
        features = features.tolist()

        # Create a response JSON object with the result
        result = {
            'latent': latent,
            'adj': adj,
            'features': features,
            'image': image,
            'status': True
        }
        return Response(result)

    # Return an error response for unsupported request methods
    return Response({'error': 'Invalid request method'})

# def process_image(request):
#     if request.method == 'POST' and request.FILES.get('image'):
#         image_file = request.FILES['image']
#         image = Image.open(image_file)
        
#         # Resize the image
#         # resized_image = image.resize((600,600))
#         # path="https://mymodernmet.com/wp/wp-content/uploads/2019/09/100k-ai-faces-6.jpg"
#         latent = gae_for(image)
#         result_list = latent.tolist()
#         # print(latent)
#         # Save the resized image to the server
#         # resized_image.save('/absolute/path/to/save/image.jpg')
        
#         # Return a JSON response indicating success
#         return JsonResponse({'message': result_list})
    
#     # If the request method is not POST or no image was provided, return an error message
#     return JsonResponse({'error': 'Invalid request.'}, status=400)

