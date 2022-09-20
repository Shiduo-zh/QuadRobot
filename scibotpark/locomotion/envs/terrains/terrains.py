import pybullet as p
import pybullet_data
import numpy as np
import random
import time
from pybullet_utils import bullet_client
from multiprocessing import Process

from scibotpark.locomotion.envs.terrains.base import Obstacle, Creator
from scibotpark.locomotion.envs.envs import UnitreeForwardEnv
from scibotpark.locomotion.envs.terrains.heightfield import HeightfieldBase

def getTerrainCls(type:str):
    info = dict()
    if type == 'moving':
        return MovingObstacle, info
    elif type == 'thin':
        return ThinObstacle, info
    elif type == 'wide':
        return WideObstacle, info
    elif type == 'hill':
        return Hill, info

class ThinObstacle(Obstacle, HeightfieldBase):
    def __init__(self,pb_client):
        super(ThinObstacle, self).__init__(pb_client)
        HeightfieldBase.__init__(self, pb_client)
        self.controller = Creator(pb_client)
        self.obstacle_id = list()
        self.bonus_id = list()
        self.bonus_num=0
        

    def create_obstacle(self):
        for i in range(6):
            self.obstacle_id.append(self.controller.create_box([0.5,0.5,2],[5*(i+1)+0.1*random.randrange(-20,20,2), -4 + 0.1 * random.randrange(-10,10,2),2],10000))
            self.obstacle_id.append(self.controller.create_box([0.5,0.5,2],[5*(i+1)+0.1*random.randrange(-20,20,2), -1 + 0.1 * random.randrange(-10,10,2),2],10000))
            self.obstacle_id.append(self.controller.create_box([0.5,0.5,2],[5*(i+1)+0.1*random.randrange(-20,20,2),  1 + 0.1 * random.randrange(-10,10,2),2],10000))
            self.obstacle_id.append(self.controller.create_box([0.5,0.5,2],[5*(i+1)+0.1*random.randrange(-20,20,2),  4 + 0.1 * random.randrange(-10,10,2),2],10000))

        #self.bonus_id=np.append(self.bonus_id,self.controller.create_ball(0.2,[0,0.3,0.3],0))
        # for i in range(6):
        #     self.bonus_id=np.append(self.bonus_id,self.controller.create_ball(0.2,[5*(i+1)+0.1*random.randrange(20,25,1),0.1*random.randrange(-70,70,2),0.3],0))
        #     self.bonus_id=np.append(self.bonus_id,self.controller.create_ball(0.2,[5*(i+1)+0.1*random.randrange(25,30,1),0.1*random.randrange(-70,70,2),0.3],0))
        
    
    def add_bonus(self,id):
        if(id!=-1):
            if (id in self.bonus_id):
                self.bonus_num+=1
                p.removeBody(id)
                print('get bonus!current bonus num is',self.bonus_num)
                return 1
            else:
                #print('no bonus!current bonus num is',self.bonus_num)
                return 0        
        else:
            return 0
    
    def setup(self):
        self.init_env()
        self.create_obstacle()
        self.init_heightfield()

class WideObstacle(Obstacle, HeightfieldBase):
    def __init__(self,pb_client):
        super(WideObstacle ,self).__init__(pb_client)
        HeightfieldBase.__init__(self, pb_client)
        self.controller = Creator(pb_client)
        self.obstacle_id = list()
        self.bonus_id = list()
        self.bonus_num = 0
        
    
    def init_env(self):
        for i in range(5):
            self.controller.create_box([5,0.2,2],[i*10,-6,2],10000)
            self.controller.create_box([5,0.2,2],[i*10,6,2],10000)
    
    def create_obstacle(self):
        for i in range(6):
            self.obstacle_id.append(self.controller.create_box([0.5,1.5,0.5],[6*(i+1)+0.1*random.randrange(-20,20,1),-2.5+0.1*random.randrange(-10,10,1),0.5],10000))
            self.obstacle_id.append(self.controller.create_box([0.5,1.5,0.5],[6*(i+1)+0.1*random.randrange(-20,20,1),2.5+0.1*random.randrange(-10,10,1),0.5],10000))
    
    def setup(self):
        self.init_env()
        self.create_obstacle()
        self.init_heightfield()

class MovingObstacle(Obstacle, HeightfieldBase):
    def __init__(self,pb_client):
        super(MovingObstacle,self).__init__(pb_client)
        HeightfieldBase.__init__(self, pb_client)
        self.controller=Creator(pb_client)
        #obstacles move in x aixs
        self.obstacle_id_x = list()
        #obstacles move in y aixs
        self.obstacle_id_y = list()

        self.basic_xs = list()
        self.basic_ys = list()
        
        
    def change_position_x(self, id, low_x, high_x, x = 0, direction = 'left'):
        current_state = self.pb_client.getBasePositionAndOrientation(int(id))
        pos = current_state[0]
        orn = current_state[1]
        dir = direction
        if(direction == 'right'):
            if(pos[0] >= high_x):
                dir='left'
                newpos = [pos[0]-x,pos[1],pos[2]]
            else:
                newpos = [pos[0]+x,pos[1],pos[2]]

        elif(direction == 'left'):
            if(pos[0] <= low_x):
                dir='right'
                newpos = [pos[0]+x,pos[1],pos[2]]
            else:
                newpos = [pos[0]-x,pos[1],pos[2]]
        
        p.resetBasePositionAndOrientation(int(id),newpos,orn)
        return dir

    def change_position_y(self,id,low_y,high_y,y=0,direction='forward'):
        current_state=self.pb_client.getBasePositionAndOrientation(int(id))
        pos=current_state[0]
        orn=current_state[1]
        if(direction=='forward'):
            if(pos[1]>=high_y):
                direction='backward'
                newpos=[pos[0],pos[1]-y,pos[2]]
            else:
                newpos=[pos[0],pos[1]+y,pos[2]]

        elif(direction=='backward'):
            if(pos[1]<=low_y):
                direction='forward'
                newpos=[pos[0],pos[1]+y,pos[2]]
            else:
                newpos=[pos[0],pos[1]-y,pos[2]]
        
        self.pb_client.resetBasePositionAndOrientation(int(id),newpos,orn)
        return direction

    def create_obstacle(self):
        shape=[0.5,0.5,2]
        #init obstacle moving in y-aixs
        for j in range(3):
            for i in range(4):
                basic_y=4*(j-1)+0.1*random.randrange(-10,10,1)
                self.basic_ys=np.append(self.basic_ys,basic_y)
                self.obstacle_id_y=np.append(self.obstacle_id_y,self.controller.create_box( shape,
                                                                                            [4+8*i+0.1*random.randrange(-10,10,1),basic_y,2],
                                                                                            10000))
        for j in range(3):
        #init obstacles moving in x-aixs
            for i in range(3):
                basic_x=8*(j+1)+0.1*random.randrange(-10,10,1)
                self.basic_xs=np.append(self.basic_xs,basic_x)
                self.obstacle_id_x=np.append(self.obstacle_id_x,self.controller.create_box( shape,
                                                                                           [basic_x,4*(i-1)+0.1*random.randrange(-10,10,1),2],
                                                                                            10000))
        self.process = Process(target = self.start_moving)

    def start_moving(self): 
         #dict initial
        dir_x=dict()
        dir_y=dict()
        for id in self.obstacle_id_x:
            newdir=random.randint(0,1)
            if(newdir==1):
                newdir='left'
            else:
                newdir='right'
            dir_x[str(id)]=newdir
        for id in self.obstacle_id_y:
            newdir=random.randint(0,1)
            if(newdir==1):
                newdir='forward'
            else:
                newdir='backward'
            dir_y[str(id)]=newdir
        #obstacle moving
        while True:
            for (id,basic_x) in zip(self.obstacle_id_x,self.basic_xs):
                dirx=self.change_position_x(id,basic_x-1,basic_x+1,0.1,dir_x[str(id)])
                dir_x[str(id)]=dirx
            for (id,basic_y) in zip(self.obstacle_id_y,self.basic_ys):
                diry=self.change_position_y(id,basic_y-1,basic_y+1,0.04,dir_y[str(id)])
                dir_y[str(id)]=diry
            time.sleep(1/20)
    
    def setup(self):
        self.init_env()
        self.create_obstacle()
        self.init_heightfield()

class Hill():
    def __init__(self,rows,cols,file_img,pb_client):
        #scale of heightmap
        self.heightRows=rows
        self.heightCols=cols
        self.heightData=np.array([0]*self.heightCols*self.heightRows).reshape((self.heightCols,self.heightRows))
        self.terrain=file_img
        self.pb_client=pb_client
        self.controller=Creator(pb_client)
        self.final_id=None

    #form random heightfield
    def random_height(self):
        for i in range(self.heightRows):
            for j in range(self.heightCols):
                self.heightData[i][j]=2*random.random()
        self.heightData=list(self.heightData.reshape((self.heightCols*self.heightRows),1))
    
    #save the proper map  data
    def save(self):
        with open('terrain//heightmaps//heightdata.txt','wb') as f:
            f.write(self.heightData)
        f.close()
    
    def form_heightmap(self):
        terrainShape = self.pb_client.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.1,.1,24],fileName = self.terrain)
        textureId = self.pb_client.loadTexture("..//heightmaps/Textures.png")
        terrain  = self.pb_client.createMultiBody(0, terrainShape)
        self.pb_client.changeVisualShape(terrain, -1, textureUniqueId = textureId)
    
    def set_final(self):
        self.final_id=self.controller.create_ball(2,[10,3,-2],0,[1,0,0,1])
    
    def set_start(self):
        self.controller.create_ball(1,[0,-25,-4],0,[0,1,1,1])
    
    def setup(self):
        self.form_heightmap()
        self.set_final()
        self.set_start()

    
if __name__=='__main__':
        from time import sleep

        ops = 'moving'

        pb_client= bullet_client.BulletClient(connection_mode= p.GUI)    
        pb_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        pb_client.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        pb_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        pb_client.setGravity(0, 0, -10)
        pb_client.setRealTimeSimulation(0)

        env = UnitreeForwardEnv(
            include_vision_obs= False, # no obs_type
            forward_reward_ratio= 1,
            alive_reward_ratio= 1,
            torque_reward_ratio= 1,
            alive_height_range= [0.2, 0.6],
            robot_kwargs= dict(
                robot_type= "a1",
                pb_control_mode= "DELTA_POSITION_CONTROL",
                pb_control_kwargs= dict(forces= [40] * 12),
                simulate_timestep= 1./500,
                default_base_transform= [0, 0, 0.42, 0, 0, 0, 1],
            ),
            pb_client= pb_client,
            nsubsteps = 10
    )
    
        
        if ops == 'hill':
            terrain='..//heightmaps//terrain2.png'
            op=Hill(1024,1024,terrain,pb_client)
            op.form_heightmap()
            op.set_final()
            op.set_start()
        
        elif ops == 'moving':
            surround = MovingObstacle(pb_client)
            surround.init_env()
            surround.create_obstacle()
            surround.process.start()
            # surround.init_heightfield()
            
        elif ops == 'thin':
            surround = ThinObstacle(pb_client)
            surround.init_env()
            surround.create_obstacle()
            surround.init_heightfield()
        
        elif ops == 'wide':
            surround = WideObstacle(pb_client)
            surround.init_env()
            surround.create_obstacle()
            surround.init_heightfield()
        
        while True:
            pb_client.stepSimulation()
            sleep(1 / 240)