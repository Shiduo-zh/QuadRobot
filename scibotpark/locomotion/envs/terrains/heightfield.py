import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
import random

numHeightfieldRows=512
numHeightfieldColumns=128

class HeightfieldBase():
    def __init__(self,
                pb_client,
                height_range=0.05,
                ):
        super(HeightfieldBase, self).__init__()
        self.pb_client=pb_client
        self.height_range=height_range
    
    def init_heightfield(self,num_rows=None,num_columns=None):
        if num_rows == None:
            num_rows = numHeightfieldRows
        if num_columns == None:
            num_columns = numHeightfieldColumns

   
        self.heightfieldData = [0] * num_rows * num_columns

        heightPerturbationRange = self.height_range
        for j in range(int(num_columns / 2)):
            for i in range(int(num_rows / 2)):
                height = random.uniform(0, heightPerturbationRange)
                self.heightfieldData[2 * i +
                                    2 * j * num_rows] = height
                self.heightfieldData[2 * i + 1 +
                                    2 * j * num_rows] = height
                self.heightfieldData[2 * i + (2 * j + 1) *
                                    num_rows] = height
                self.heightfieldData[2 * i + 1 + (2 * j + 1) *
                                    num_rows] = height
        for j in range(-5, 5):
            for i in range(-5, 5):
                x = int(num_rows / 4) + i
                y = int(num_columns / 4) + j
                # print(x, y)
                self.heightfieldData[2 * x +
                                    2 * y * num_rows] = 0
                self.heightfieldData[2 * x + 1 +
                                    2 * y * num_rows] = 0
                self.heightfieldData[2 * x + (2 * y + 1) *
                                    num_rows] = 0
                self.heightfieldData[2 * x + 1 + (2 * y + 1) *
                                    num_rows] = 0


        self.terrainShape = self.pb_client.createCollisionShape(
                shapeType=self.pb_client.GEOM_HEIGHTFIELD,
                meshScale=[.12, .12, 1.0],
                heightfieldTextureScaling=0,
                heightfieldData=self.heightfieldData,
                numHeightfieldRows=num_rows,
                numHeightfieldColumns=num_columns)
        terrain = self.pb_client.createMultiBody(0, self.terrainShape,basePosition=[256,0,0])
        self.pb_client.resetBasePositionAndOrientation(terrain, [0.0, 0.0, 0.0], [0, 0, 0, 1])

        self.pb_client.changeVisualShape(terrain,
                                        -1,
                                        rgbaColor=[0.1, 0.1, 0.1, 1],
                                        specularColor=[0.1, 0.1, 0.1, 1])