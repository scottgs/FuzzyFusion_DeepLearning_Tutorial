CREATE TABLE frame (
    id SERIAL PRIMARY KEY,
    image TEXT NOT NULL, -- image file name: use ";" to separate multiple associated images
    class_label INT, -- class label: may use binary one hot encoding in multi-label case
    partitions hstore -- denormalized partitioning information
);

CREATE INDEX idx_frame ON frame USING GIN (partitions);

CREATE TABLE partitions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(32) UNIQUE NOT NULL, -- name of the partition namespace
    labels VARCHAR[] NOT NULL, -- names of the partitions that exist in the namespace
    created_on TIMESTAMP NOT NULL,
    reference TEXT, -- reference to the associated data shard
    summary BOOLEAN NOT NULL, -- required to show in the summary
    impl_version INT -- data structure impl: v1: hstore tagging {"part" => 1}, v2: hstore enum {"namespace"=>"part"}
);

CREATE TABLE class_label (
    id SERIAL PRIMARY KEY, -- do not use a foreign key to reference this optional table because it could be NULL or N/A
    label_namespace TEXT, -- namespace for multiple datasets
    label_name TEXT NOT NULL, -- name of the label
    label_name_long TEXT, -- long name of the label
    label_id INT NOT NULL -- label id
);

CREATE TABLE xval_metrics (
    model_id TEXT NOT NULL, -- a unique name/id of the model e.g. res50.5fold::A
    subset_id INT NOT NULL, -- can be class_label if the concept applies
    partition_name TEXT NOT NULL, -- aka namespace of the fold name
    fold_name TEXT NOT NULL,
    metrics hstore, -- denormalized metrics
    PRIMARY KEY (model_id, subset_id, partition_name, fold_name)
);

CREATE TABLE inference (
    frame_id INT REFERENCES frame(id) NOT NULL, -- a reference to the frame
    model_id TEXT NOT NULL, -- a unique name/id of the model e.g. res50.5fold::A
    prediction hstore NOT NULL, -- all outputs from model inference
    PRIMARY KEY (frame_id, model_id)
);
