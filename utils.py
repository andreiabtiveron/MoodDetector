import matplotlib.pyplot as plt

# funções auxiliares para visualização e análise
def show_image(img):
    plt.imshow(img.reshape(48,48), cmap="gray")
    plt.axis("off")
    plt.show()
