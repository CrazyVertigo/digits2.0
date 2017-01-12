close all; clear;
caffe_root = '../../';
addpath(genpath([caffe_root,'matlab/']));
use_gpu = 0;


save_root = '../../data/fcn_label_full/results/';
if ~exist(save_root, 'dir'), mkdir(save_root); end;
% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end
test_imgs_dir = '../../data/fcn_label_full/test/';
model_dir = './';
%model_dir = '../../examples/finetune/';
net_model = [model_dir 'fcn11_deploy.prototxt'];
net_weights = [model_dir 'fcn11_full_iter_15000.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end
% Initialize a network
net = caffe.Net(net_model, net_weights, phase);
%get data
test_imgs = dir([test_imgs_dir,'*.jpg']);
test_imgs = {test_imgs.name};
for i = 1:length(test_imgs),test_imgs{i} = test_imgs{i}(1:end-4);end
for i = 1:length(test_imgs)
    im = imread([test_imgs_dir,test_imgs{i},'.jpg']);
    siz=size(im);
    blobdata = net.blob_vec(net.name2blob_index('data'));
    oldshape=blobdata.shape;
    newshape=[siz(2),siz(1),oldshape(3),oldshape(4)];
    blobdata.reshape(newshape);
    im_data = im(:, :, [3, 2, 1]);   % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);       % convert from uint8 to single
    scores = net.forward({im_data});
    s = scores{1};
    figure(1);
    subplot(3, 4, 1); imshow(im);
    for j = 1:10
       subplot(3, 4, j + 1);
       imshow(s(:,:,j)');
       title(num2str(j - 1));
    end
    colormap('hot');
    print(1, '-dpng', [save_root, test_imgs{i}, '_hot']);
    s_max = s(:, :, 1:10);
    s_max(s_max < 0.5) = 0;
    [s_max, idx_max] = max(s_max, [], 3);
    s_max = s_max'; idx_max = idx_max';
    [L, num] = bwlabel(s_max, 8);
    idx_results = cell(num, 1);
    r_means = cell(num, 1);
    c_means = cell(num, 1);
    for l = 1:num
        [r_tmp, c_tmp] = find(L == l);
        if length(r_tmp) < 10
            continue;
        end
        s_tmp = s_max(L == l);
        w_tmp = s_tmp./sum(s_tmp);
        r_mean = double(uint32(sum(w_tmp.*r_tmp)));
        c_mean = double(uint32(sum(w_tmp.*c_tmp)));        
% %         r_mean = double(uint32(mean(r_tmp)));
% %         c_mean = double(uint32(mean(c_tmp)));
        idx_results{l} = idx_max(r_mean, c_mean) - 1;
        r_means{l} = r_mean; c_means{l} = c_mean;
    end
    idx_results = cell2mat(idx_results); r_means = cell2mat(r_means); c_means = cell2mat(c_means);
    
    figure(2); imshow(im); hold on;
    for l = 1:length(idx_results)
        text(c_means(l), r_means(l), num2str(idx_results(l)), 'FontSize', 50, 'FontWeight', 'demi', 'Color', 'r', 'BackgroundColor', 'g');
    end

    print(2, '-dpng', [save_root, test_imgs{i}]);
    save([save_root, test_imgs{i}, '.mat'], 's', 'idx_results', 'r_means', 'c_means');
    close( figure(1));
    close( figure(2));
end
caffe.reset_all();
