
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import ipywidgets as widgets
from IPython.display import display


check_point_stability = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
#torch_dtype=torch.float32 si le GPU n'est pas disponible
#torch_dtype=torch.float16 si le GPU est disponible
pipe = StableDiffusionPipeline.from_pretrained(check_point_stability, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")


check_point_fon_translate = "kingabzpro/Helsinki-NLP-opus-yor-mul-en"
tokenizer = AutoTokenizer.from_pretrained(check_point_fon_translate)
model = AutoModelForSeq2SeqLM.from_pretrained(check_point_fon_translate)



"""phrase_yoruba = "Mo wa nibi ni irú"
encodage = tokenizer(phrase_yoruba, return_tensors="pt")  # conversion en tenseur PyTorch

#Predict translate
with torch.no_grad():
  traduction = model.generate(**encodage, max_length=64, num_beams=2)  # on ajuste les paramètres

traduction_dechiffree = tokenizer.batch_decode(traduction, skip_special_tokens=True)[0]
print(traduction_dechiffree)

#Predict image
image = pipe(traduction_dechiffree).images[0]

image.save("new_img.png")
"""

def translate_yor_en(phrase_yoruba):
  encodage = tokenizer(phrase_yoruba, return_tensors="pt")  # conversion en tenseur PyTorch

  #Predict translate
  with torch.no_grad():
    traduction = model.generate(**encodage, max_length=64, num_beams=2)  # on ajuste les paramètres

  traduction_dechiffree = tokenizer.batch_decode(traduction, skip_special_tokens=True)[0]
  return traduction_dechiffree

def gen_img_yor_text(phrase_yoruba):
  traduction_dechiffree = translate_yor_en(phrase_yoruba)
  #Predict image
  image = pipe(traduction_dechiffree).images[0]

  image.save("new_img.png")
"""
phrase_yoruba = "Aworan ti ile-iwe pẹlu awọn ọmọ ile-iwe"
translate_sentent = translate_yor_en(phrase_yoruba)
print(translate_sentent)

gen_img_yor_text(phrase_yoruba)
"""

# Define font and colors
LARGE_FONT = ("Verdana", 16)
BUTTON_FONT = ("Verdana", 12)
FONT_COLOR = "#333333"
BG_COLOR = "#F2F2F2"
BUTTON_COLOR = "#007bff"
BUTTON_HOVER_COLOR = "#0056b3"
ENTRY_BORDER_COLOR = "#ccc"

# Initialize the main window
root = widgets.Output()

# Define header title
header_label = widgets.Label(value="HACKATHON IA 2024", layout=widgets.Layout(margin='20px 0px 0px 0px'), style={'description_width': 'initial'})
display(header_label)

# Define text input area
text_entry = widgets.Textarea(layout=widgets.Layout(width='50%', height='150px', margin='10px 0px 0px 0px'))
display(text_entry)

# Define translation buttons
yoruba_to_french_button = widgets.Button(
    description="Yorouba en Français", layout=widgets.Layout(width='50%', margin='10px 0px 0px 0px'), button_style='info'
)
display(yoruba_to_french_button)

fon_to_french_button = widgets.Button(
    description="Fon en Français", layout=widgets.Layout(width='50%', margin='10px 0px 0px 20px'), button_style='info'
)
display(fon_to_french_button)

# Define image generation buttons
yoruba_image_button = widgets.Button(
    description="Générer image par Yoruba", layout=widgets.Layout(width='50%', margin='10px 0px 0px 0px'), button_style='info'
)
display(yoruba_image_button)

fon_image_button = widgets.Button(
    description="Générer image par Fon", layout=widgets.Layout(width='50%', margin='10px 0px 0px 20px'), button_style='info'
)
display(fon_image_button)

# Define horizontal separator line
separator_line = widgets.Output()
with separator_line:
    print('----------------------------------------------------------------------------------------')
display(separator_line)

# Define text result area
text_result = widgets.Label(value="", layout=widgets.Layout(width='50%', height='100px', margin='10px 0px 0px 0px'))
display(text_result)

def translate_text(language):
    input_text = text_entry.value
    # Implement translation logic based on the selected language (yoruba or fon)
    translated_text = translate_yor_en(input_text)
    text_result.value = f"Traduction française: {translated_text}"

def generate_image(language):
    input_text = text_entry.value
    gen_img_yor_text(input_text)
    # Implement image generation logic based on the selected language (yoruba or fon)
    # Display the generated image or show a message box if image generation is successful or not

yoruba_to_french_button.on_click(lambda button: translate_text("yoruba"))
fon_to_french_button.on_click(lambda button: translate_text("fon"))
yoruba_image_button.on_click(lambda button: generate_image("yoruba"))
fon_image_button.on_click(lambda button: generate_image("fon"))

display(root)