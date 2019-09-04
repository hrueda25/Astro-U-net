from __future__ import division
import time
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import sys
from astropy.stats import biweight_location
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats
from datetime import datetime
from datetime import timedelta
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(3)   # The id of your GPU


# folders names change them according yours
# -------------------------------
folder_num = str('Network1')
# create_image_noise = False
input_dir = './dataset/in_data_poisson/'   # noisy images
gt_dir = './dataset/out_data/'             # ground truth
checkpoint_dir = './astro_result' + folder_num + '/'
result_dir = './astro_result' + folder_num + '/'   # results folder
test_net = './astro_result' + folder_num + '/test_net/'
name_txt_file = result_dir + "test" + folder_num + ".txt"
path_val = "./dataset/validation/"
path_noise_val = "./dataset/validation/noise/"

# -------------------------------


# other parameters
# -------------------------------
ps = 256  # patch size for training
save_freq = 500
ron = 3   # e
dk = 7   # e/hr/pix;
# -------------------------------

if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels, exp_time=None, exp=False):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
    if not exp:
        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])
    if exp:
        cons = tf.fill(tf.shape(deconv), exp_time)
        c = tf.slice(cons, [0, 0, 0, 0], [-1, -1, -1, 1])
        deconv_output = tf.concat([deconv, x2, c], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2 + 1])
    return deconv_output


def network(input, e):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512, exp_time=e, exp=True)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 1, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    return conv10


def fits_read(name):
    f = fits.open(name)
    if "EXPTIME" in f[0].header:
        e = f[0].header["EXPTIME"]
    else:
        e = 0
    return e


def save_fits(image, name, path):
    hdu = fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path + name + '.fits')


def squeeze_img(array, fil, ps=256):
    """change dimension of network output"""
    out = np.squeeze(array, axis=0)
    o = np.zeros((ps, ps, 1))
    o[:, :, 0] = out[:, :, fil]
    out = np.squeeze(o, axis=2)
    return out


def valid_img(epoch, cnt):
    """test network during training"""
    if not os.path.isdir(test_net + '%04d' % epoch):
        os.makedirs(test_net + '%04d' % epoch)
    path = "./dataset/validation/noise/"
    data_list = glob.glob(path + "*.fits")
    v_num = np.random.randint(0, len(data_list) - 6)
    for name in data_list[v_num: v_num+1]:
        one_image(name, test_net + '%04d/' % epoch, cnt)
    print("validation done")


def gen_image(img, fil, i, j, exp_time=None, ps=256, end_i=False, end_j=False):
    i_end = i + ps
    j_end = j + ps
    if end_i:
        i_end = None
    if end_j:
        j_end = None
    in_imag = img[i: i_end, j: j_end]
    in_img = np.zeros((ps, ps, 1))
    in_img[:, :, 0] = in_imag
    in_patch = np.expand_dims(in_img, axis=0)
    output = sess.run(out_image, feed_dict={in_image: in_patch, ex: 2})
    output = squeeze_img(output, 0)
    return output


def one_image(name, sv_path, cnt):
    """iterate whole image, predict output and make one big image"""
    ps = 256
    fil = 0
    img = fits.getdata(name, ext=0)
    step = 256
    div = ps/step
    W, H = img.shape[0: 2]
    w_diff = (ps - (W % 256))
    w_c = ps + int(np.ceil(w_diff/2))
    w_f = ps + int(np.floor(w_diff/2))
    h_diff = (ps - (H % 256))
    h_c = ps + int(np.ceil(h_diff/2))
    h_f = ps + int(np.floor(h_diff/2))
    w = W + w_c+w_f
    h = H + h_c+h_f
    image = np.zeros((w, h, 1))
    image[w_c: w_c + W, h_c: h_c + H, 0] = img
    output = np.zeros((w, h))
    print(w/ps, h/ps)
    for i in range(0, w, step):
        for j in range(0, h, step):
            in_patch = image[i: i+ps, j: j+ps]
            in_patch = np.expand_dims(in_patch, axis=0)
            new_image = sess.run(out_image, feed_dict={in_image: in_patch, ex: 2})
            output[i:i+ps, j: j+ps] += new_image[0, :, :, 0]
    output = (output/div**2)[w_c: w_c + W, h_c:h_c + H]
    save_fits(output, name[27:-5] + "_" + str(cnt) + "_output", sv_path)


def text_write(file, epoch, cnt, loss, time, name):
    """save parameters into text file"""
    file = open(file, "a")
    file.write(str(epoch) + "\t" + str(cnt) + "\t" + str(loss) + "\t" + str(time) + "\t" + str(name) + "\n")
    file.close


def black_level(arr, max_num, level=0.1):
    """Prevent to have an image with more than some percentage of zeroes as input
       level - percentage <0,1>; 0.1/10% default"""
    arr = list(np.hstack(arr))
    per = arr.count(0.00000000e+00)/len(arr)
    if max_num > 10:
        level = 0.3
    if per < level or max_num > 15:
        return True
    else:
        return False


def poisson_noise(name, ratio=2, xx=None, yy=None, ps=None, ron=3, dk=7, sv_name=None, path=None, patch_noise=False, ret=True, save=False):
    """add noise into image"""
    img_data = fits.getdata(name, ext=0)   # e/sec
    if patch_noise:
        img_data = img_data[xx:xx + ps, yy:yy + ps]
    width, height = img_data.shape[0:2]
    exp_time = int(name[-10:-5])
    img = img_data * exp_time   # e
    DN = np.random.poisson(np.sqrt(dk*exp_time/ratio), (width, height))
    RON = np.random.poisson(ron, (width, height))
    SN = np.random.poisson(np.sqrt(img/ratio))
    noise_img = (img/ratio + SN + RON + DN)/(exp_time/ratio)
    noise_img = np.where(img_data == 0.00000000e+00, 0.00000000e+00, noise_img)
    if save:
        save_fits(noise_img, sv_name, path)
    if ret:
        return noise_img

sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 1])
gt_image = tf.placeholder(tf.float32, [None, None, None, 1])
ex = tf.placeholder(tf.float32, name="TIME")
out_image = network(in_image, ex)


G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
t_vars = tf.trainable_variables()

lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
g_loss = []
data_list = glob.glob(gt_dir + "*.fits")
allfolders = glob.glob('./astro_result' + folder_num + '/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))


learning_rate = 1e-4
for epoch in range(lastepoch, 8001):
    num = 0
    if os.path.isdir('./astro_result' + folder_num + '/%04d' % epoch):
        continue
    cnt = 0
    if epoch > 2000:
        learning_rate = 1e-5

    for name in np.random.permutation(data_list):
        st = time.time()
        num += 1
        out = fits.getdata(name, ext=0)
        ex_time = float(int(name[-10:-5])//2)   # exposure time is in the name of images, also in fits header

        H, W = out.shape[0], out.shape[1]
        zero_level = False
        max_num = 0
        while not zero_level:
            xx = np.random.randint(0, H - ps)
            yy = np.random.randint(0, W - ps)
            arr = out[xx:xx + ps, yy:yy + ps]
            zero_level = black_level(arr, max_num)
            max_num += 1
        # ground truth
        out_img = np.zeros((ps, ps, 1))
        out_img[:, :, 0] = out[xx:xx + ps, yy:yy + ps]
        gt_patch = np.expand_dims(out_img, axis=0)

        # input image
        in_name = glob.glob(input_dir + name[19:-11] + "*.fits")
        in_ = fits.getdata(in_name[0], ext=0)
        in_imag = in_[xx:xx + ps, yy:yy + ps]
        del in_
        in_img = np.zeros((ps, ps, 1))
        in_img[:, :, 0] = in_imag
        in_patch = np.expand_dims(in_img, axis=0)

        cnt += 1
        _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                        feed_dict={in_image: in_patch, ex: 2, gt_image: gt_patch, lr: learning_rate})

        g_loss.append(G_current)
        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss), time.time() - st))

        if cnt % 50 == 0:
            text_write(name_txt_file, epoch, cnt, np.mean(G_current), time.time() - st, name[19:])

        if epoch % save_freq == 0 and cnt % 50 == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)
            sv_name = "/" + name[19:-5]
            output = squeeze_img(output, 0)
            save_fits(out[xx:xx + ps, yy:yy + ps], sv_name, result_dir + '%04d' % epoch)
            save_fits(output, sv_name + "_output", result_dir + '%04d' % epoch)
            save_fits(in_imag, sv_name + "_noise", result_dir + '%04d' % epoch)
        if epoch % (save_freq*3) == 0 and cnt % 100 == 0:
            valid_img(epoch, cnt)
    saver.save(sess, checkpoint_dir + 'model.ckpt')
    if epoch >= 1000 and epoch % (save_freq*2) == 0:
        saver.save(sess, checkpoint_dir + 'model%04d.ckpt' % epoch)

