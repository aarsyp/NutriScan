import os
import dotenv

from PIL import Image
from langsmith.wrappers import wrap_openai
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langsmith import traceable
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import streamlit as st
import io
import openai

# load the .env file
dotenv.load_dotenv()

# get OPENAI_API_KEY from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Konfigurasi Azure (ganti dengan informasi Anda)
subscription_key = "ebc59ab8dca44c8abd5b4c66cdcc3a4c"
endpoint = "https://nutrieye.cognitiveservices.azure.com/"

# Inisialisasi klien Azure Computer Vision
credentials = CognitiveServicesCredentials(subscription_key)
client = ComputerVisionClient(endpoint, credentials)

def read_text_from_image(image_stream):
    try:
        read_response = client.read_in_stream(image_stream, raw=True)
        read_operation_location = read_response.headers["Operation-Location"]

        operation_id = read_operation_location.split("/")[-1]
        while True:
            read_result = client.get_read_result(operation_id)
            if read_result.status not in ["notStarted", "running"]:
                break

        if read_result.status == OperationStatusCodes.succeeded:
            extracted_text = ""
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    extracted_text += line.text + "\n"
            return extracted_text
        else:
            return "OCR gagal."
    except Exception as e:
        return f"Terjadi kesalahan saat membaca teks dari gambar: {e}"

st.title("NutriScan")

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Tampilkan gambar
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        # Lakukan OCR
        with io.BytesIO(uploaded_file.getvalue()) as image_stream:
            extracted_text = read_text_from_image(image_stream)

        st.subheader("Hasil Output:")
        #st.write(extracted_text)

        ocr_output = extracted_text

        text_prompt = """
        Context:
        Anda adalah seorang peneliti nutrisi yang sedang mengembangkan sebuah aplikasi untuk membantu orang dalam memahami fakta gizi dari berbagai makanan. Aplikasi ini akan menyediakan ringkasan tentang informasi nutrisi dari makanan berdasarkan input pengguna.

        Task:
        Buatlah ringkasan dari nutrition fact yang mudah dipahami dan berikan saran tentang mana yang baik untuk dikonsumsi dan tidak berdasarkan nilai gizi.

        Instruction:
        Analisis nilai gizi dari nutrition fact, termasuk kalori, lemak, karbohidrat, protein, gula, vitamin, dan mineral.
        Apabila ada informasi tentang takaran saji, bisa sesuaikan lagi informasinya.
        Identifikasi nilai gizi yang sesuai dengan kebutuhan dan gaya hidup pengguna.
        Buatlah ringkasan yang jelas dan informatif, dengan fokus pada nilai gizi yang penting.
        Berikan saran tentang mana yang baik untuk dikonsumsi dan tidak berdasarkan nilai gizi, dengan mempertimbangkan kebutuhan dan gaya hidup pengguna.
        Tawarkan next action yang dapat dilakukan pengguna, seperti mencari resep masakan sehat atau mencari informasi lebih lanjut tentang nutrisi tertentu.

        Clarify:
        Apakah ada informasi gizi tertentu yang lebih penting bagi pengguna?
        Apakah pengguna memiliki alergi atau pantangan makanan tertentu?
        Apakah pengguna memiliki tujuan kesehatan tertentu, seperti menurunkan berat badan atau meningkatkan kebugaran?
        Refine:

        Gunakan bahasa yang mudah dipahami dan hindari istilah teknis yang rumit.
        Sajikan informasi dalam format yang menarik dan mudah dibaca, seperti tabel atau grafik.
        Berikan contoh konkret tentang bagaimana pengguna dapat menerapkan saran yang diberikan.

        Data:
        {}
        """

        # prompt_template = HumanMessagePromptTemplate.from_template(text_prompt)

        openaiclient = wrap_openai(openai.Client(api_key="sk-proj-0JvJCTzTgl4N4fKJCUBxT3BlbkFJTOL9VpVUhhJ3cdbx5D8v"))

        @traceable
        def pipeline(user_input: str):
            result = openaiclient.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Anda adalah seorang peneliti nutrisi yang sedang mengembangkan "
                            "sebuah aplikasi untuk membantu orang dalam memahami fakta gizi dari "
                            "berbagai makanan. Aplikasi ini akan menyediakan ringkasan tentang "
                            "informasi nutrisi dari makanan berdasarkan input pengguna."
                        ),
                    },
                    {
                        "role": "user",
                        "content": text_prompt.format(user_input),
                    }
                ],
            )
            return result.choices[0].message.content

        summary = pipeline(user_input=ocr_output)

        st.write(summary)
    except Exception as e:
        st.error(f"Terjadi kesalahan pada aplikasi: {e}")
