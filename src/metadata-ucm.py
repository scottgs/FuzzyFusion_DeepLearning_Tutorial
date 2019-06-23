import itertools
import pandas as pd
import psycopg2, psycopg2.extras
import h5py
from pathlib import Path
from imeta import *
DATASET = '/mnt/datasets/IEEE/UCMerced'
DATABASE = 'dbname=ucmerced user=postgres'

def build_index(conn):
    print("Initializing metadata...")
    cur = conn.cursor()
    dataset = Path(DATASET)
    class_dirs = [x for x in dataset.iterdir() if x.is_dir()]
    labels = sorted(map(lambda pth: str(pth.relative_to(dataset)), class_dirs))
    sql = 'INSERT INTO frame (image, class_label, partitions) values %s'
    values = []
    for class_dir in class_dirs:
        for fname_image in class_dir.iterdir():
            label = str(class_dir.relative_to(dataset))
            label_int = labels.index(label)
            partitions = ','.join(['"active"=>1'] + ['"{}"=>1'.format(label), '"part"=>"{}"'.format(label)]) # TODO + ['"traintest"=>"train"'])
            values.append(( str(fname_image.relative_to(dataset)), label_int, partitions))
    psycopg2.extras.execute_values(cur, sql, values)
    sql = """INSERT INTO partitions (name, labels, created_on, summary, impl_version) values
        ('active', array['active'], now(), true, 1),
        ('part', array{}, now(), false, 1);""".format(labels)
    cur.execute(sql)
    conn.commit()
    add_xval(cur, 5, '5fold')
    conn.commit()
    add_xval(cur, 10, '10fold')
    conn.commit()
    psycopg2.extras.execute_values(cur, 'INSERT INTO class_label (label_id, label_name) values %s', list(enumerate(labels)))
    conn.commit()

def use_downsample_v1(frac_downsample):
    with psycopg2.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("""UPDATE frame SET partitions = delete(partitions, 'active');""")
        cur.execute("""UPDATE frame SET partitions = partitions || '"active"=>1' WHERE random()<%f;""", (frac_downsample, ))
        conn.commit()

def use_downsample_v2(name, frac_downsample):
    with psycopg2.connect(DATABASE) as conn:
        cur = conn.cursor()
        cur.execute("""UPDATE frame SET partitions = delete(partitions, %s);""", (name, ))
        cur.execute("""UPDATE frame SET partitions = partitions || '"%s"=>1' WHERE random()<%f;""" % (name, frac_downsample))
        cur.execute("""INSERT INTO partitions (name, labels, created_on, summary, impl_version) values ('%s', array['%s'], now(), true, 1)
            ON CONFLICT DO NOTHING;""" % (name, name))
        conn.commit()

if __name__ == "__main__":
    metadata(DATABASE, build_index)
    # use_downsample_v2('down10', 0.1)
    # use_downsample_v2('down20', 0.2)
