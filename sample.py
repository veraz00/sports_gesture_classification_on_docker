def train(args):
    # Validate input arguments
    if not isinstance(args, dict):
        raise ValueError("args must be a dictionary")
    
    required_keys = ["fold", "pretrained_model", "parent_dataset_dir", "csv_filename", "numpy_test"]
    for key in required_keys:
        if key not in args:
            raise ValueError(f"args is missing required key: {key}")

    # Load the CSV file
    df = pd.read_csv(os.path.join(args["parent_dataset_dir"], args["csv_filename"]))

    # Create dictionary mapping labels to indices
    target_dict = {vv: i for i, vv in enumerate(df.labels.unique())}
    with open('label.json', 'w') as ff:
        json.dump(target_dict, ff)
    df['labels_index'] = df['labels'].apply(lambda x: target_dict[x])

    # Select train data
    df = df[df['data set'] == 'train']

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set hyperparameters
    num_epochs = 50
    batch_sizes = {'train': 8, 'valid': 8}
    best_acc_score = 0.6
    saved_path = 'Densenet121_sports_classification.pt'

    # Split data into train and validation sets
    df_train = df[df.kfold != args["fold"]].reset_index(drop=True)
    df_valid = df[df.kfold == args["fold"]].reset_index(drop=True)

    train_images = df_train.filepaths.values.tolist()
    train_images = [os.path.join(args["parent_dataset_dir"], i) for i in train_images]
    train_targets = df_train.labels_index.values

    valid_images = df_valid.filepaths.values.tolist()
    valid_images = [os.path.join(args["parent_dataset_dir"], i) for i in valid_images]
    valid_targets = df_valid.labels_index.values

    # Create data loaders
    train_datasets = Custom_dataset(image_list=train_images, label_list=train_targets, transform=image_transforms['train'])
    valid_datasets = Custom_dataset(image_list=valid_images, label_list=valid_targets, transform=image_transforms['valid'])
    data_loaders = {
        'train': DataLoader(train_datasets, batch_size=batch_sizes['train'], shuffle=True),
        'valid': DataLoader(valid_datasets, batch_size=batch_sizes['valid'], shuffle=False)
    }

    # Initialize model and optimizer
    model = DenseNet121(num_class=len(target_dict)).to(device)    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max"
    )

    # Load pretrained model if specified
    if args["pretrained
