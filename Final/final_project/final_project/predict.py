import tensorflow as tf
import numpy as np
import os
from PIL import Image
import pandas as pd

result = []
sess = tf.Session()
new_saver = tf.train.import_meta_graph('./tfmodel/mnist-20000 .meta')
new_saver.restore(sess, './tfmodel/mnist-20000 ')
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
prob = graph.get_tensor_by_name("Placeholder:0")
for i in range(9): 
    result.append([]) 
for file in os.listdir('./img/'):
    img=Image.open('./img/' + file).convert('L')
    array=np.asarray(img,dtype="float32")
    array = array.reshape((1,784))
    name = file.split('.')
    order = name[0].split('_')
    feed_dict ={x:array, prob:1.0}
    op_to_restore = graph.get_tensor_by_name("cf_op:0")
    p = sess.run(op_to_restore,feed_dict)
    result[int(order[0])].insert(int(order[1]), np.argmax(p))
number = []
for i in range(len(result)):
    s1 = result[i]
    s2 = [str(j) for j in result[i]]
    s3 = ''.join(s2)
    number.append(s3)
output = pd.DataFrame({'number':number})
print(output)
output.to_csv('result.csv', index = False)