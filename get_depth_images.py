from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests

from os import listdir
from os.path import isfile, join


options = Options()
options.headless = True

driver = webdriver.Chrome(chrome_options=options)

path = "./datasets/tiny_nerf/images"
image_paths = [f for f in listdir(path) if isfile(join(path, f))]

driver.get("https://3dphoto.io/uploader/index.html")

for image_path in image_paths:

    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME , "my_field"))).send_keys(f"D:/codefiles/NeRF-kaedim/datasets/tiny_nerf/images/{image_path}")

    driver.find_element(By.NAME , "Submit").click()

    image_src = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "img"))).get_attribute("src")

    print(image_src)

    img_data = requests.get(image_src).content

    with open(f"./datasets/tiny_nerf/depths/{image_path}", "wb") as f:
        f.write(img_data)

    driver.refresh()

# driver.quit()