model = dict(
    type = 'm2det',
    input_size = 512,
    init_net = True,
    pretrained = '/media/stmoon/Data/droneeye_weight/M2Det_VIS_size512_netvgg16_epoch70.pth', #vgg16_reducedfc.pth',
    m2det_config = dict(
        backbone = 'vgg16',
        net_family = 'vgg', # vgg includes ['vgg16','vgg19'], res includes ['resnetxxx','resnextxxx']
        base_out = [22,34], # [22,34] for vgg, [2,4] or [3,4] for res families
        planes = 256,
        num_levels = 8,
        num_scales = 6,
        sfam = False,
        smooth = True,
        num_classes =  12, #81
        ),
    rgb_means = (104, 117, 123),
    p = 0.6,
    anchor_config = dict(
        step_pattern = [8, 16, 32, 64, 128, 256],
        size_pattern = [0.06, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        ),
    save_eposhs = 10,
    weights_save = 'weights/'
    )

train_cfg = dict(
    cuda = True,
    warmup = 5,
    per_batch_size = 4,     # 16
    lr = [0.004, 0.002, 0.0004, 0.00004, 0.000004],
    gamma = 0.1,
    end_lr = 1e-6,
    step_lr = dict(
        COCO = [90, 110, 130, 150, 160],
        VOC = [100, 150, 200, 250, 300], # unsolve
        VIS = [100, 150, 200, 250, 300], # TODO: check it
        ),
    print_epochs = 1,
    num_workers= 8,
    )

test_cfg = dict(
    cuda = True,
    topk = 0,
    iou = 0.45,
    soft_nms = True,
    score_threshold = 0.1,
    keep_per_class = 50,
    save_folder = 'eval'
    )

loss = dict(overlap_thresh = 0.5,
            prior_for_matching = True,
            bkg_label = 0,
            neg_mining = True,
            neg_pos = 3,
            neg_overlap = 0.5,
            encode_target = False)

optimizer = dict(type='SGD', momentum=0.9, weight_decay=0.0005)

dataset = dict(
    VIS = dict(
        train_sets = ['VisDrone2019-VID-train'],
        eval_sets = ['VisDrone2019-VID-val-rot'],
        ),
    VOC = dict(
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')],
        eval_sets = [('2007', 'test')],
        ),
    COCO = dict(
        train_sets = [('2014', 'train'), ('2014', 'valminusminival')],
        eval_sets = [('2014', 'val')],
        test_sets = [('2015', 'test-dev')],
        )
    )

import os
home = os.path.expanduser("/media/stmoon/Data/")
VOCroot = os.path.join(home,"data/VOCdevkit/")
COCOroot = os.path.join(home,"coco/")
VISroot = os.path.join(home,"VisDrone/Task2_Object_Detection_in_Videos/")

