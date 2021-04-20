img_in_name = '../curiosity_medium.png';
kernel_name = '../../kernel/kernel_medium.png';
img_out_name = 'blur_edgetaper.png';

rgb = imread(img_in_name);

psf = double(imread(kernel_name));
psf = psf/sum(psf(:));


edgesTapered = edgetaper(rgb,psf);

imwrite(edgesTapered, img_out_name);

