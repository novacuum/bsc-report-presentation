import inspect, pandas, numpy, re
import operator
from hashlib import sha1
from pathlib import Path

from document.report.scripts.image import get_pipeline_from_result
from engine.helpers import write_file, read_file
from engine.nn.training import NNModel
from model.testing import TestingResult
from utils.function import get_functions
from utils.report import create_sorted_result_set

TARGET_DIR = Path('../table/generated')

column_format_result_overview = """l
S[table-format=2.1]@{\,\( \pm \)\,}
S[table-format=1.1, table-number-alignment = left]
S[table-format=2.2]@{\,\( \pm \)\,}
S[table-format=1.2]
"""


def accuracy(x):
    return '%2.1f' % (x * 100)


def loss(x):
    return '%1.2f' % x


def create_result_overview_table_from_data(data):
    with pandas.option_context("max_colwidth", 1000, 'display.precision', 2):
        clean_up_pattern = re.compile(r'^.*_remove_.*$', re.MULTILINE)
        remove_cite = re.compile(r'\\cite{[^}]*}', re.MULTILINE)

        formatters = numpy.repeat([accuracy, loss], 2)
        col_index = pandas.MultiIndex.from_tuples([('Model', '', '')])
        col_index = col_index.append(pandas.MultiIndex.from_product([['Test'], ['Accuracy (\si{\percent})', 'Loss'], ['_remove_', 'it']]))
        tex = pandas.DataFrame(data, columns=col_index) \
            .to_latex(escape=False, index=False, column_format=column_format_result_overview, formatters=numpy.concatenate(([None], formatters)).tolist())
        return remove_cite.sub('', clean_up_pattern.sub('', tex))\
            .replace('\gls{xpps}', 'xpps').replace('\gls{hog}', 'HOG').replace('\gls{mcl}', 'mcl').replace('\gls{nrs}', 'nrs')


def create_result_overview_table(test_name, *include):
    data = []
    for result in create_sorted_result_set(test_name):
        row = TestingResult(result).get_stat_table_row(*include)
        data.append([row[i] for i in [0, 5, 6, 7, 8]])
    return create_result_overview_table_from_data(data)


def table_result_overview_sct_compressed():
    return create_result_overview_table('testing_sct_compressed', 'features', 'nr_sensitivity')


def table_result_overview_sct_vl():
    return create_result_overview_table('testing_sct_vl', 'features', 'spectrogram')


def table_result_overview_sct_padded():
    return create_result_overview_table('testing_sct_left_padded', 'features', 'spectrogram', 'nr_sensitivity')


def table_result_overview_scs():
    data = []
    for result in create_sorted_result_set('testing_scs_r3'):
        if '_raw_' in result['id']:
            row = TestingResult(result).get_stat_table_row('features', 'split')
            data.append([row[i] for i in [0, 5, 6, 7, 8]])
    return create_result_overview_table_from_data(data)

#
# def table_result_overview_scs_all():
#     data = [TestingResult(result).get_stat_table_row('features', 'split') for result in create_sorted_result_set('testing_scs_r3')]
#     return create_result_overview_table_from_data(data)
#
#
# def table_result_overview_sct():
#     data = [TestingResult(result).get_stat_table_row('features', 'nr_sensitivity') for result in create_sorted_result_set('testing_sct_compressed')][0:5]
#     data += [TestingResult(result).get_stat_table_row('features', 'spectrogram') for result in create_sorted_result_set('testing_sct_vl')][0:5]
#     data += [TestingResult(result).get_stat_table_row('features', 'spectrogram', 'nr_sensitivity') for result in create_sorted_result_set('testing_sct_left_padded')][0:5]
#     for i, row in enumerate(data):
#         data[i][0] = row[0].split(',')[0] + ', ' + ('compressed' if i < 5 else ('variable length' if i < 10 else 'left padded')) + "," + row[0].split(',')[-1]
#
#     data = sorted(data, key=operator.itemgetter(5, 2), reverse=True)
#     return create_result_overview_table_from_data(data)


def run(forced=None):
    TARGET_DIR.mkdir(exist_ok=True)
    forced = {} if forced is None else set(forced)

    for f, fx in get_functions(__name__, 'table_'):
        name = f[6:]
        dest = TARGET_DIR / f'{name}.tex'

        digest = sha1(inspect.getsource(fx).encode()).hexdigest()
        digest_path = Path(f'{dest}.sha1')

        if f in forced or not dest.is_file() or not (digest_path.is_file() and digest == read_file(digest_path)):
            print(f'Generating {dest}...')
            write_file(dest, fx())
            write_file(digest_path, digest)
        else:
            print(f'Skipped {dest}')
    print('Done')


if __name__ == '__main__':
    run()
