#coding:utf-8

import caffe
import cv2
import numpy as np
from scipy.spatial.distance import pdist
import mxnet as mx
import numpy as np
from PIL import Image
from collections import namedtuple
import time
net = caffe.Net('../../mobilefacenet_official_128/mobilefacenet.prototxt', '../../mobilefacenet_official_128/mobilefacenet.caffemodel',caffe.TEST)
def caffeGetFeature(imgPath):
	bgr = cv2.imread(imgPath)
	#cv2.imshow("BGR",img)
	#cv2.waitKey(0)

	# BGR 0 1 2
	# RGB 2 
	rgb = bgr[...,::-1]
	rgb = (rgb - 128.0) / 128.0
	#rgb = rgb.transpose((2,1,0)) 
	rgb = np.swapaxes(rgb, 0, 2)
	rgb = np.swapaxes(rgb, 1, 2) 
	rgb = rgb[None,:] # add singleton dimension
	#cv2.imshow("RGB",rgb)
	#cv2.waitKey(0)
	#print (rgb)
	out = net.forward_all( data = rgb ) # out is probability
	#print(out['fc1'][0])
	a = out['fc1'][0]
	return a

##########################MXnet##########



 
         
#读取一张本地图片
def read_one_img(img_path):
    #这里注意是jpg，即3通道rgb，如果不是的话需要转换
    #img=Image.open(img_path)
    img = cv2.imread(img_path,1)
    #img=img.resize((112,112),Image.BILINEAR)
    #img=np.array(img)
    #return np.array([[img[:,:,i] for i in xrange(3)]])
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2) 
    Batch = namedtuple('Batch', ['data'])
    #这里也要吐槽，数据要封装成这样的形式才能用
    one_data=Batch([mx.nd.array(np.array([img]))])
    return one_data
 
def mxnetForward(img_path):
    #数据的shape，这样写有点复杂了，这里吐槽，哈哈
    data_shape=[('data', (1,3,112,112))]
     
    #读取一张图片，路径替换成你的图片路径
    data=read_one_img(img_path)
  
    #装载模型,epoch为Inception-BN-0039.params的数字
    prefix="../../mobilefacenet_official_128/model"
    model=mx.module.Module.load(prefix=prefix,epoch=0,context=mx.cpu())
     
    #这里bind后就可以直接用了，不需要再定义网络架构（这个比tensorflow简洁）
    model.bind(data_shapes=data_shape)
     
    #前馈过程
    #model.forward(data,is_train=False)
     
    #获取前馈过程的输出(这个设计为何？)，result类型为list([mxnet.ndarray.NDArray])
    #result = model.get_outputs()
     
    #输出numpy类型的数组
    #result=result[0].asnumpy()
     
    #下面取最大值作为预测值 
    #pos=np.argmax(result)
    ##print 'max:',np.max(result)
    #print 'position:',pos
    #print "result:",clazz_map[pos]
     
     
    #获取网络结构
    internals=model.symbol.get_internals()
    #print '\n'.join(internals.list_outputs())
     
    #获取从输入到flatten层的特征值层的模型
    feature_net = internals["fc1_output"]
    feature_model=mx.module.Module(symbol=feature_net,context=mx.cpu())
    feature_model.bind(data_shapes=data_shape)
     
    #获取模型参数
    arg_params,aux_params=model.get_params()
     
    #上面只是定义了结构，这里设置特征模型的参数为Inception_BN的
    feature_model.set_params(arg_params=arg_params,
                             aux_params=aux_params, 
                             allow_missing=True)
     
    #输出特征
    feature_model.forward(data,is_train=False)
    feature = feature_model.get_outputs()[0].asnumpy()
     
    print 'shape:',np.shape(feature)
    print 'feature:',feature
    return feature

# a = caffeGetFeature("../../test_images/1.jpg")
# print(a)
'''
[-9.0736228e-01  4.2107370e-01 -1.0902348e+00  1.9374862e-01
  1.8992447e+00 -1.2892399e+00 -1.3971800e+00  2.9950373e+00
 -9.3173921e-01  1.6387993e+00 -1.7985886e+00  1.6722292e-02
 -1.9071250e+00  5.7284099e-01  5.0786203e-01 -1.1543369e+00
 -1.3465647e-01 -6.9894868e-01  3.5834044e-01  4.7926661e-01
 -3.0668202e-01  1.9281213e-01 -1.3090357e+00 -1.3502263e+00
  7.0354927e-01  1.1220961e+00 -1.6119552e+00  7.6998189e-02
 -5.4537064e-01  6.1222869e-01 -5.0801024e-02  1.3425790e+00
 -5.7691641e-02 -2.9050097e-01  2.7423358e+00 -1.0105273e-01
 -4.1336378e-01  8.9381531e-02 -1.2221947e+00  1.2823175e+00
 -1.5800674e+00  7.7786070e-01 -6.0689974e-01 -7.3363519e-01
 -5.2541089e-01  3.1380004e-01  1.0787828e+00  8.2826561e-01
  3.0599761e-01 -3.7479214e-04  1.2681001e+00  6.2170309e-01
 -3.5985279e-01  6.4659178e-01 -4.3959892e-01  3.8034266e-01
 -1.8551891e-01 -8.6611640e-01 -2.9309857e-01 -3.7187237e-01
 -1.5277402e-01 -5.7730979e-01  5.0249869e-01  1.2554098e+00
 -1.2939954e+00 -1.5119395e+00 -7.5431705e-01  1.2115378e+00
  1.5224317e-01 -8.1541318e-01  1.0216969e+00 -1.2463360e+00
 -1.4328822e+00  2.2581372e+00  2.8201205e-01 -1.1968231e+00
 -7.3843759e-01  1.7313272e-01 -7.7154076e-01  1.1134810e+00
  1.3496643e-01 -9.9447119e-01  2.3307152e+00 -8.7796068e-01
 -9.1684818e-01  6.7829865e-01  7.2469217e-01 -1.0646147e+00
 -1.0934581e+00 -1.0846058e+00  1.8243586e+00 -4.3073431e-01
 -3.1350300e-01  5.4787554e-02  1.1802294e+00 -1.5309875e+00
 -2.0802774e-01  1.4789613e+00 -1.5904433e-01  3.3858833e-01
  1.5678594e-01 -1.2891765e+00 -2.1968832e-02  2.8274184e-01
 -2.0820105e+00  1.0247118e+00  9.8832238e-01  6.4990348e-01
 -7.9703444e-01  1.3682793e+00  8.3256382e-01  1.1854844e+00
  1.6676376e+00 -1.9578593e+00  7.9664439e-02  1.3808453e-01
 -8.7862098e-01 -1.7287415e-01  1.3564652e+00  9.4036371e-02
 -1.5409557e+00  8.4669179e-01 -1.2913995e+00  3.4359235e-01
  8.9586985e-01 -1.1773318e+00 -1.5270583e-01  8.7184501e-01]
'''

a = mxnetForward("../../test_images/1.jpg")
'''
[[-0.73063284  0.4547649  -1.2803757  -0.1269572   1.9413625  -1.6424401
  -1.2587032   3.1537476  -0.7099047   1.5928655  -1.504669    0.05434592
  -1.6472442   0.27408364  0.9950586  -0.40375146 -0.1389905  -0.31646866
   0.71363914  0.6708147  -0.7974488   0.448961   -1.265979   -1.4045935
   0.2899743   0.9183864  -1.3732625   0.06393101 -0.7983758   0.77812624
   0.17387284  1.2749907   0.20626357 -0.7312328   2.449217   -0.1963461
  -0.1354531   0.23003304 -1.0094417   1.3209593  -1.4001164   0.16056408
  -0.6465269  -0.32949486 -0.65216684  0.678284    1.3034259   0.7263274
   0.3457045   0.00707366  1.1885642   0.16871262 -0.4162801   0.96293545
  -1.2073991   0.2835454  -0.02571693 -1.0273663  -0.06479694 -0.3515636
  -0.1009702  -0.51476985  0.64473647  1.1062382  -0.95807874 -1.5838534
  -0.8307209   1.3604038   0.22047473 -0.983649    1.338003   -1.3450868
  -1.044248    1.8629231   0.12516715 -1.0829525  -0.627525    0.5991133
  -0.50404984  1.179976    0.24306586 -0.94048697  2.4487555  -0.82210207
  -0.49869677  0.92964405  0.35444993 -1.2052778  -0.9113398  -0.95183647
   1.5626038  -0.09671046 -0.49555483 -0.0339651   1.0550796  -2.0421488
  -0.14754973  1.8181356  -0.34780547  0.15685631  0.58857864 -1.1081443
   0.3976446   0.47585434 -1.9074157   1.5396135   1.084188    0.66860914
  -0.9666197   1.5191317   0.7107022   0.9653745   1.6970205  -1.8352735
  -0.42221305  0.00805547 -1.0386001  -0.55750155  1.5851123   0.21618384
  -1.2366602   0.88257974 -1.3738087  -0.01858817  0.92974824 -1.4122245
  -0.04248272  1.2048495 ]]
'''
# b = mxnetForward("../../test_images/2.jpg")
# a = caffeGetFeature("../../test_images/1.jpg")
# b = caffeGetFeature("../../test_images/2.jpg")
# d1=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
# d2 = 1 - pdist(np.vstack([a,b]),'cosine')
# print(d1,d2)

# counter = 0
# t1 = time.time()
# rs = []
# dir = "lfwAligned/"
# with open('pairs_1.txt') as f:
# 	lines = f.readlines()
# 	#print(len(lines))
# 	for line in lines:
# 		line = line.strip()
# 		arr = line.split(",")
# 		Limg = dir + arr[0]
# 		Rimg = dir + arr[1]
# 		#print Limg,Rimg
# 		gtoundTruthLable = arr[2]
# 		Lfeat = caffeGetFeature(Limg)
# 		Rfeat = caffeGetFeature(Rimg)
# 		#print(arr[2])
# 		cosSim =  1 - pdist(np.vstack([Lfeat,Rfeat]),'cosine')
# 		#print(gtoundTruthLable,"===>>",type(cosSim),cosSim[0])
# 		rs.append(gtoundTruthLable + "," + str(cosSim[0]))
# 		counter = counter + 1
# 		t2 = time.time()
# 		if counter % 100 ==0:
# 			print(counter,t2 - t1)
# 			t1 = t2
# 			#break
# 		#print(line)


# with open('rs.txt','w') as f:
# 	for r in rs:
# 		f.write(r + "\n")