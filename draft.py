from dataloaders import utils
train_transform = utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                        download_flag=True, transform=train_transform, 
                        seed=self.seed, rand_split=args.rand_split, validation=args.validation)