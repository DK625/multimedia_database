from rembg import remove
from PIL import Image

inp  = "D:\\a2m\\multimedia_database\\data_process_preparing\\preprocessed_images\\desert_mountains\\free-photo-of-thien-nhien-sa-m-c-d-i-qu-d-i.jpg"

outp = "D:\\a2m\\multimedia_database\\data_process_preparing\\preprocessed_images\\desert_mountains\\fhihih.jpg"

input = Image.open(inp)

output = remove(input)
output.save(outp)