from __future__ import division
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
import time
import scipy

from skimage.measure import compare_ssim as ssim
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize
from photutils import CircularAperture
from photutils import DAOStarFinder
import math
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(4)  # The id of your GPU
# weights avaible here https://drive.google.com/open?id=1I2GxwSX-yDUEJVuSgOov32-Lz81cwdiS

file_name = './image/eval.txt'
test_net = './image/'
checkpoint_dir = './weight1/'
ps = 256  # patch size for training


if not os.path.isdir(test_net):
                os.makedirs(test_net)


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


def upsample_nn_and_concat(x1, x2, output_channels, in_channels, exp_time=None, exp=False):
    up_sample = tf.image.resize_bilinear(x1, (tf.shape(x2)[1], tf.shape(x2)[2]))
    conv_up = slim.conv2d(up_sample, output_channels, [3, 3], rate=1, activation_fn=None)
    if not exp:
        deconv_output = tf.concat([conv_up, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])
    if exp:
        cons = tf.fill(tf.shape(conv_up), exp_time)
        c = tf.slice(cons, [0, 0, 0, 0], [-1, -1, -1, 1])
        deconv_output = tf.concat([conv_up, x2, c], 3)
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
    # out = tf.depth_to_space(conv10, 2)
    return conv10


def psnr(original, denoise):
    """calculate peak signal-noise ratio,
    returns MSE, relative error and PSNR"""
    mse = np.mean((original-denoise) ** 2)
    sigma_max = np.max(sigma_clip(original, sigma=3, masked=False, copy=False))
    print(sigma_max, mse)
    return  10*np.log10(sigma_max**2/mse)


def text_write(file, psnr, ssim):
    """save parameters into text file"""
    file = open(file, "a")
    file.write(str(psnr) + "\t" + str(ssim) + "\t" + "\n")
    file.close


def image_evaluation(original, denoise, file):
    psnr_test = psnr(original, denoise)
    ssim_test = ssim(original, denoise, multichannel=False)
    text_write(file, psnr_test, ssim_test)


def poisson_noise(name, exp_time=None, ratio=None, xx=None, yy=None, ps=None, ron=3, dk=7, sv_name=None, path=None, patch_noise=False, ret=True, save=False):
    img_data = fits.getdata(name, ext=0)    # e/sec
    if patch_noise:
        img_data = img_data[xx:xx + ps, yy:yy + ps]
    width, height = img_data.shape[0: 2]
    if exp_time is None:
        exp_time = int(name[-10: -5])   # sec /2 bc we want just half of exp time
    img = img_data * exp_time   # e
    DN = np.random.poisson(np.sqrt(dk*exp_time/ratio), (width, height))
    RON = np.random.poisson(ron, (width, height))
    SN = np.random.poisson(np.sqrt(np.abs(img)/ratio))
    noise_img = (img/ratio + SN + RON + DN)/(exp_time/ratio)
    noise_img = np.where(img_data == 0.00000000e+00, 0.00000000e+00, noise_img)
    if save:
        save_fits(noise_img, sv_name, path)
    if ret:
        return noise_img


def squeeze_img(array, fil, ps=256):
    """change dimension of network output"""
    out = np.squeeze(array, axis=0)
    o = np.zeros((ps, ps, 1))
    o[:, :, 0] = out[:, :, 0]
    out = np.squeeze(o, axis=2)
    return out


def fits_read(name):
    f = fits.open(name)
    if "EXPTIME" in f[0].header:
        e = f[0].header["EXPTIME"]
    else:
        e = 0
    return e


def save_fits(image, name, path, header):
    hdu = fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path + name + '.fits')


def black_level(arr, max_num, level=0.3):
    arr = list(np.hstack(arr))
    per = arr.count(0.00000000e+00)/len(arr)
    if per < level or max_num == 10:
        return True
    else:
        return False


def valid_img2(path, data_original, sv_path, img_name, xx=None, yy=None):
    """test network during training"""
    data_noise = np.sort(glob.glob(path + "*" + img_name + "*.fits"))
    data_or = np.sort(glob.glob(data_original + "*" + img_name + "*.fits"))
    for i in range(len(data_noise)):
        if xx is not None:
            one_image(data_noise[i], sv_path, xx, yy, data_or[i], save=True)
    print("validation 29 done")


def one_image(name, sv_path, xx, yy, data_original=None, file_name=None, save=False, eval_=True, small=False):
    """iterate whole image, predict output and make one big image"""
    print(name)
    ratio = 2
    ps = 256
    file_name = './image/eval.txt'
    fil = 0    # filter_name(name)
    ex_time = fits_read(data_original)
    input_image = fits.getdata(name, ext = 0)
    org, header = fits.getdata(data_original, 1, header=True)
    if small:
        in_img = np.zeros((ps, ps, 1))
        in_img[:, :, 0] = input_image[xx: xx + 256, yy: yy + 256]
        in_patch = np.expand_dims(in_img, axis=0)
        output = sess.run(out_image, feed_dict={in_image: in_patch, ex: ratio})
        output = squeeze_img(output, 0, ps=256)
    if not small:
        step = 32
        img = input_image[xx[0]: xx[1], yy[0]: yy[1]]
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
        for i in range(0, w, step):
            for j in range(0, h,  step):
                in_patch = image[i: i+ps, j: j+ps]
                in_patch = np.expand_dims(in_patch,  axis=0)
                new_image = sess.run(out_image, feed_dict={in_image: in_patch, ex: ratio})
                output[i: i+ps, j: j+ps] += new_image[0, :, :, 0]
        output = (output/(8*8))[w_c: w_c + W, h_c: h_c + H]
    if save:
        save_name = sv_path + name[27:-5] + "_" + str(xx[0]) + "_" + str(xx[1]) + "_" + str(yy[0])+"_" + str(yy[1])
        fits.writeto(save_name + '.fits', output, header)
        norm = ImageNormalize(org, interval=ZScaleInterval(), stretch=LinearStretch())
        plt.imshow(output, cmap='Greys_r', origin='lower', norm=norm)
        plt.axis('off')
        plt.savefig(save_name + ".png", dpi=600, bbox_inches='tight', pad_inches=0)
    if eval_:
        image_evaluation(org[xx[0]: xx[1], yy[0]: yy[1]], output, file_name)

sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 1])
gt_image = tf.placeholder(tf.float32, [None, None, None, 1])
ex = tf.placeholder(tf.float32, name="TIME")
out_image = network(in_image, ex)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
print("Neural network will rule the world!!")
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
saver.restore(sess, ckpt.model_checkpoint_path)

path = "./dataset/validation/noise/"
path_original = "./dataset/validation/"

x = 0
y = 0
st1 = time.time()
for img_name in ['11360_e1', '11700_2t', '11730_04', '11700_93', '11700_51', '11730_15']:
    if img_name == '11360_e1':
        l = [[4100, 5000, 2200, 3750], [2400, 3100, 800, 2000]]
        for i in range(len(l)):
            x += -l[i][0] + l[i][1]
            y += -l[i][2] + l[i][3]
            xx = [l[i][0], l[i][1]]
            yy = [l[i][2], l[i][3]]
            valid_img2(path, path_original, test_net, img_name, xx, yy)
    if img_name == '11700_2t':
        l = [[2000, 3000, 1400, 2800],  [2250, 4000, 3700, 4500]]
        for i in range(len(l)):
            x += -l[i][0] + l[i][1]
            y += -l[i][2] + l[i][3]
            st = time.time()
            xx = [l[i][0], l[i][1]]
            yy = [l[i][2], l[i][3]]
            valid_img2(path, path_original, test_net, img_name, xx, yy)
            print(time.time()-st)
    if img_name == '11730_04':
        l = [[1200, 2480, 1800, 2568], [1900, 2900, 2800, 4000]]
        for i in range(len(l)):
            x += -l[i][0] + l[i][1]
            y += -l[i][2] + l[i][3]
            st = time.time()
            xx = [l[i][0], l[i][1]]
            yy = [l[i][2], l[i][3]]
            valid_img2(path, path_original, test_net, img_name, xx, yy)
            print(time.time()-st)
    if img_name == '11700_93':
        l = [[2600, 4200, 880, 2200]]
        for i in range(len(l)):
            x += -l[i][0] + l[i][1]
            y += -l[i][2] + l[i][3]
            xx = [l[i][0], l[i][1]]
            yy = [l[i][2], l[i][3]]
            valid_img2(path, path_original, test_net, img_name, xx, yy)
    if img_name == '11700_51':
        l = [[2400, 4100, 300, 4100]]
        for i in range(len(l)):
            x += -l[i][0] + l[i][1]
            y += -l[i][2] + l[i][3]
            st = time.time()
            xx = [l[i][0], l[i][1]]
            yy = [l[i][2], l[i][3]]
            valid_img2(path, path_original, test_net, img_name, xx, yy)
            print(time.time()-st)
    if img_name == '11730_15':
        l = [[1100, 2200, 1600, 2800], [2800, 4100, 1200, 2400]]
        for i in range(len(l)):
            x += -l[i][0] + l[i][1]
            y += -l[i][2] + l[i][3]
            st = time.time()
            xx = [l[i][0], l[i][1]]
            yy = [l[i][2], l[i][3]]
            valid_img2(path, path_original, test_net, img_name, xx, yy)
            print(time.time()-st)

print(x/10, y/10, (time.time()-st1)/10)
exit()

