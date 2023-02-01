import inspect, json, re, pandas, numpy, random
import math
import shutil
from argparse import ArgumentParser
from hashlib import sha1
from pathlib import Path

from engine.helpers import write_file, read_file
from model.report import ReportResult
from plot import operator
from plot.inputdescriptor import InputDescriptor, create_descriptor_set
from plot.operator import ResultTaskPropGetter, DictMapper, ModelMatcher, FeatureMatcher, FunctionMapper, Mapper
from plot.scatter import create_scatter
from plot.settings import create_color_map_for_symbols
from utils.experiment import load_audio_and_db
from utils.function import get_functions

from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
import matplotlib.image as mpimg

from utils.report import create_sorted_result_set

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Tex Gyre Pagella']
rcParams['svg.fonttype'] = 'none'
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Helvetica'] # Tahoma
rcParams['font.size'] = 10
rcParams['image.cmap'] = 'viridis'

FIGURES_DIR = Path('../image/generated')


def get_all_test_results():
    data = create_sorted_result_set('testing_sct_compressed')
    data += create_sorted_result_set('testing_sct_vl')
    data += create_sorted_result_set('testing_sct_left_padded')
    return data


def create_input_preview(result_name, horizontal=True):
    sorted_results = create_sorted_result_set(result_name)
    result_model = ReportResult(sorted_results[0], result_name)
    image_paths = list(map(lambda f: f.path(), result_model.spectrogram_images.files))
    rows = len(result_model.labels)
    cols = 5

    if horizontal:
        rows = math.ceil(rows / 2)
        cols = 10

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14, rows*2), sharex=True, sharey=True)

    for ax in axes.ravel().tolist():
        ax.set_xticks([]), ax.set_yticks([])

    for row, label in enumerate(result_model.labels):
        filtered_image_paths = list(filter(lambda image_path: image_path.find('_' + label + '.') != -1, image_paths))
        random_filtered_image_paths = random.sample(filtered_image_paths, 5)
        if horizontal:
            y = math.floor(row/2)
            offset = (row % 2) * 5
        else:
            y = row
            offset = 0
            # axes[y][offset].set_ylabel(label)

        for i in range(5):
            image = mpimg.imread(random_filtered_image_paths[i])
            ax = axes[y][offset + i]
            ax.axis("off")
            ax.set_title(label)
            ax.imshow(image, aspect='equal', cmap='afmhot_r', interpolation='antialiased')

    return fig


def create_model_distribution_per_result(result_name, sort=None, **descriptors: InputDescriptor):
    return create_scatter(plt, create_sorted_result_set(result_name), descriptors, sort)


def sort_model_distribution_sct_compressed(args):
    index = numpy.argsort([int(v) for v in args['x']])
    for arg_key in args:
        args[arg_key] = numpy.array(args[arg_key])[index].tolist()


def sort_model_distribution_sct_variable_length(args):
    index = numpy.lexsort((numpy.array(args["s"])*-1, args['x']))
    for arg_key in args:
        args[arg_key] = numpy.array(args[arg_key])[index].tolist()

    args.pop('s')
    print(args.keys())


def image_width_resolution_distribution(path):
    descriptors = create_descriptor_set(
        y=InputDescriptor('Spectrogram width', ResultTaskPropGetter('CreateSpectrogramTask', 'x_pixels_per_sec', 2000, DictMapper({2000: '2000', 4000: '4000', 5000: '5000'})))
        , x=InputDescriptor('Test accuracy', operator.itemgetter('test_acc_m'))
        , c=InputDescriptor('Model', ModelMatcher())
        , markers=InputDescriptor('Features', FeatureMatcher())
    )
    return create_scatter(plt, get_all_test_results(), descriptors, horizontal=True)


def image_height_resolution_distribution(path):
    descriptors = create_descriptor_set(
        y=InputDescriptor('Spectrogram height', ResultTaskPropGetter('CreateSpectrogramTask', 'height', 256, DictMapper({256: '256', 300: '300', 512: '512'})))
        , x=InputDescriptor('Test accuracy', operator.itemgetter('test_acc_m'))
        , c=InputDescriptor('Model', ModelMatcher())
        , markers=InputDescriptor('Features', FeatureMatcher())
    )
    return create_scatter(plt, get_all_test_results(), descriptors, horizontal=True)


def image_model_distribution_sct_compressed(path):
    return create_model_distribution_per_result(
        'testing_sct_compressed'
        , x=InputDescriptor('Sensitivity', ResultTaskPropGetter('NoiseReduceTask', 'sensitivity', 0, DictMapper({0:'0', 6:'6', 12:'12', 24:'24'})))
        , y=InputDescriptor('Test accuracy', operator.itemgetter('test_acc_m'))
        , c=InputDescriptor('Model', ModelMatcher())
        , markers=InputDescriptor('Features', FeatureMatcher())
        , sort=sort_model_distribution_sct_compressed
    )


def image_model_distribution_sct_variable_length(path):
    xpps_color_map = create_color_map_for_symbols([2000, 5000, 4000])
    return create_model_distribution_per_result(
        'testing_sct_vl'
        , x=InputDescriptor('Spectrogram height', ResultTaskPropGetter('CreateSpectrogramTask', 'height', 0, DictMapper({256:'256', 300:'300', 512:'512'})))
        , y=InputDescriptor('Test accuracy', operator.itemgetter('test_acc_m'))
        , c=InputDescriptor('Pixel per sec.', ResultTaskPropGetter('CreateSpectrogramTask', 'x_pixels_per_sec', 0, DictMapper(xpps_color_map)))
        , s=InputDescriptor('Pixel per sec.', ResultTaskPropGetter('CreateSpectrogramTask', 'x_pixels_per_sec', 0, DictMapper({2000:40, 4000:41, 5000:42})))
        , markers=InputDescriptor('Features', FeatureMatcher())
        , sort=sort_model_distribution_sct_variable_length
    )


def image_model_distribution_sct_padded(path):
    return create_model_distribution_per_result(
        'testing_sct_left_padded'
        , x=InputDescriptor('Model', ModelMatcher(Mapper()))
        , y=InputDescriptor('Test accuracy', operator.itemgetter('test_acc_m'))
        , c=InputDescriptor('Spectrogram height', ResultTaskPropGetter('CreateSpectrogramTask', 'height', 0, FunctionMapper(lambda v: v / 256)))
        , s=InputDescriptor('Sensitivity', ResultTaskPropGetter('NoiseReduceTask', 'sensitivity', 0, FunctionMapper(lambda v: v * 5 + 15)))
        , markers=InputDescriptor('Features', FeatureMatcher())
    )


def image_model_distribution_scs_r3(path):
    return create_model_distribution_per_result(
        'testing_scs_r3'
        , x=InputDescriptor('Model', ModelMatcher(Mapper()))
        , y=InputDescriptor('Test accuracy', operator.itemgetter('test_acc_m'))
        , c=InputDescriptor('Cover area', ResultTaskPropGetter('SplitIntoPartsTask', 'label_min_cover_length', 0))
        , s=InputDescriptor('Strides', ResultTaskPropGetter('SplitIntoPartsTask', 'strides', 0, FunctionMapper(lambda v: v * 7500)))
        , markers=InputDescriptor('Features', FeatureMatcher())
    )


def image_compressed_spectorgram_b2(path):
    ppl = load_audio_and_db('simple_call_test').extract_label_parts(False)
    ppl.files = [file for file in ppl.files if file.metadata.duration > .2]

    spectrogram_file_list = ppl.create_spectrogram(sampling_rate=500000, width=100, window='Ham').run()
    shutil.copy(spectrogram_file_list.files[0].path(), str(path).replace('.svg', '_compressed.png'))

    # when 22ms = 100pixel, we get 4545,454545 pixel per sec
    spectrogram_file_list = ppl.create_spectrogram(sampling_rate=500000, x_pixels_per_sec=4545, window='Ham').run()
    shutil.copy(spectrogram_file_list.files[0].path(), str(path).replace('.svg', '.png'))


def image_spectrogram_preview_variable_length(path):
    return create_input_preview('testing_sct_vl')


def image_spectrogram_preview_padded(path):
    return create_input_preview('testing_sct_left_padded', False)


def run(forced=None):
    FIGURES_DIR.mkdir(exist_ok=True)
    forced = {} if forced is None else set(forced)

    for f, fx in get_functions(__name__, 'image_'):
        name = f[6:]
        dest = FIGURES_DIR / f'{name}.svg'

        digest = sha1(inspect.getsource(fx).encode()).hexdigest()
        digest_path = Path(f'{dest}.sha1')

        if f in forced or not (digest_path.is_file() and digest == read_file(digest_path)):
            print(f'Generating {dest}...')

            plot = fx(dest)
            if plot:
                plot.savefig(dest, bbox_inches='tight', dpi=300)

            write_file(digest_path, digest)
        else:
            print(f'Skipped {dest}')
    print('Done')


if __name__ == '__main__':
    parser = ArgumentParser(description='Generate diagrams for the report')
    parser.add_argument('forced', nargs='*', help='images that must be regenerated even when source is unchanged')
    args = parser.parse_args()
    run(**vars(args))
