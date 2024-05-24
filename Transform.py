import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from scipy.stats import norm
from math import floor

class Image:

    def __init__(self, image:np.ndarray, transformImage:np.ndarray):
        self.__image:np.ndarray = image
        self.__transforImage:np.ndarray = transformImage

    def plot(self):
        _, axes = plt.subplots(1, 2)
        axes[0].imshow(self.__image, cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        axes[1].imshow(self.__transforImage, cmap='gray')
        axes[1].set_title('Imagen Transformada')
        axes[1].axis('off')
        plt.show()

class ImageTransfor:

    @staticmethod
    def loadImage(path:str)->np.ndarray:
        return io.imread(path)

    def __init__(self, mean:int=0, std:int=1, imageBounds:tuple[int]=(28, 28), size:int=20, seed:int=1)->None:
        self.__imageList:list[np.ndarray] = []
        self.__imageBounds:tuple[int] = imageBounds
        self.distNorm = norm(loc=mean, scale=std)
        self.__size = size
        self.__seed = seed

    def __transforImage(self, image:np.ndarray)->np.ndarray:
        # if self.__imageBounds != image.shape:
        #     raise "las figuras no coinciden"
        imagenTranformada = image.copy()
        randPointsX = self.distNorm.rvs(size=self.__size, random_state=self.__seed)%image.shape[1]
        randPointsY = self.distNorm.rvs(size=self.__size, random_state=self.__seed+1)%image.shape[0]
        for i in range(self.__size):
            x, y = randPointsX[i], randPointsY[i]
            imagenTranformada[floor(y)][floor(x)] = 1.0
        return imagenTranformada
        

    def append(self, image:np.ndarray)->None:
        self.__imageList.append(Image(image, self.__transforImage(image)))
    
    def getAllImage(self)->np.ndarray:
        return np.array(self.__imageList)
    
    def __getitem__(self, index):
        return self.__imageList[index]
    
    def __len__(self):
        return len(self.__imageList)
    
    def __iter__(self):
        return iter(self.__imageList)

t = ImageTransfor(imageBounds=(224, 225), mean=112, std=30, size=1000)
t.append(io.imread("./images/seven.png"))
t.append(io.imread("./images/seven.jpg"))
for i in t:
    i.plot()