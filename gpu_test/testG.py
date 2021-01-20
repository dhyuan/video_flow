import tensorflow as tf


tf.debugging.set_log_device_placement(True)

a = tf.constant([1.2,2.3,3.6], shape=[3],name='a')
b = tf.constant([1.2,2.3,3.6], shape=[3],name='b')
 
c = a+b
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(session.run(c))

print('--------')
print('is_gpu_available %s ' % tf.test.is_gpu_available())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

