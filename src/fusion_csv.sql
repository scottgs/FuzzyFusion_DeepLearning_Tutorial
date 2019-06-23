-- Usage: sudo -u postgres psql ucmerced -f fusion_csv.sql

-- export cross validation accuracies (per fold per class)
\copy (SELECT fold_name, subset_id AS class_label, metrics -> 'acc' AS accuracy FROM xval_metrics WHERE model_id LIKE 'res50.%' AND partition_name = '5fold' ORDER BY fold_name, subset_id) TO '/tmp/xval_cls.res50.5fold.csv' WITH CSV HEADER DELIMITER ',';

-- export average cross validation accuracies (per fold)
\copy (SELECT fold_name, AVG((metrics -> 'acc')::float) AS accuracy FROM xval_metrics WHERE model_id LIKE 'res50.%' AND partition_name = '5fold' GROUP BY fold_name ORDER BY fold_name) TO '/tmp/xval.res50.5fold.csv' WITH CSV HEADER DELIMITER ',';

-- export inference data by model
\copy (SELECT partitions -> '5fold' AS fold_name, image, class_label AS y_true, (prediction -> 'y_pred')::int AS y_pred, prediction -> 'y_vec' as y_vec FROM inference y LEFT JOIN frame x ON y.frame_id = x.id WHERE model_id LIKE 'res50.5fold::%' ORDER BY partitions -> '5fold') TO '/tmp/pred.res50.5fold.csv' WITH CSV;

-- export inference data by model & fold
\copy (SELECT image, class_label AS y_true, (prediction -> 'y_pred')::int AS y_pred, prediction -> 'y_vec' as y_vec FROM inference y LEFT JOIN frame x ON y.frame_id = x.id WHERE model_id = 'res50.5fold::A' ORDER BY partitions -> '5fold') TO '/tmp/pred.res50.5fold::A.csv' WITH CSV;
\copy (SELECT image, class_label AS y_true, (prediction -> 'y_pred')::int AS y_pred, prediction -> 'y_vec' as y_vec FROM inference y LEFT JOIN frame x ON y.frame_id = x.id WHERE model_id = 'res50.5fold::B' ORDER BY partitions -> '5fold') TO '/tmp/pred.res50.5fold::B.csv' WITH CSV;
\copy (SELECT image, class_label AS y_true, (prediction -> 'y_pred')::int AS y_pred, prediction -> 'y_vec' as y_vec FROM inference y LEFT JOIN frame x ON y.frame_id = x.id WHERE model_id = 'res50.5fold::C' ORDER BY partitions -> '5fold') TO '/tmp/pred.res50.5fold::C.csv' WITH CSV;
\copy (SELECT image, class_label AS y_true, (prediction -> 'y_pred')::int AS y_pred, prediction -> 'y_vec' as y_vec FROM inference y LEFT JOIN frame x ON y.frame_id = x.id WHERE model_id = 'res50.5fold::D' ORDER BY partitions -> '5fold') TO '/tmp/pred.res50.5fold::D.csv' WITH CSV;
\copy (SELECT image, class_label AS y_true, (prediction -> 'y_pred')::int AS y_pred, prediction -> 'y_vec' as y_vec FROM inference y LEFT JOIN frame x ON y.frame_id = x.id WHERE model_id = 'res50.5fold::E' ORDER BY partitions -> '5fold') TO '/tmp/pred.res50.5fold::E.csv' WITH CSV;
