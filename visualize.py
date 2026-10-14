"""
Plots of the training runs that the training scripts record in metrics/*.json (see `utils.save_metrics`).

notebooks/Visualize.ipynb calls the public functions below. They all take `metrics_dir`: None searches the usual
places (see `find_metrics_dirs`), a path or a list of paths reads only those folders. The JSON files are read again on
every call, so re-running a notebook cell while a script is still training shows the epochs finished since.

Only numpy, pandas, matplotlib, scikit-learn and (optionally) IPython are imported - not torch, transformers or the
other modules of the repo - so this file also runs where the training dependencies are not installed.
"""

import json
import os
import textwrap
import warnings
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

TASKS = ('sentiment', 'paraphrase', 'sonnet')
KAGGLE_WORKING = Path('/kaggle/working')
KAGGLE_INPUT = Path('/kaggle/input')

# Class names by label id, checked against example sentences of data/ids-{sst,cfimdb}-dev.csv.
SENTIMENT_CLASSES = {
    'sst': ['very negative', 'negative', 'neutral', 'positive', 'very positive'],
    'cfimdb': ['negative', 'positive'],
}
DATASET_NAMES = {'sst': 'SST-5', 'cfimdb': 'CFIMDB'}
SENTIMENT_MODES = ('last-linear-layer', 'full-model')  # the choices of classifier.py --fine-tune-mode
PARAPHRASE_CLASSES = ['not paraphrase', 'paraphrase']

# Light-mode reference palette of the dataviz guidelines. Series hues are used in this fixed order and stay with their
# entity (a fine-tuning mode, a run) whatever else is drawn; the confusion matrices use the single-hue blue ramp.
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
MODE_COLORS = dict(zip(SENTIMENT_MODES, SERIES_COLORS))
BLUE_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
             '#1c5cab', '#184f95', '#104281', '#0d366b']
CONFUSION_CMAP = LinearSegmentedColormap.from_list('blue_ramp', BLUE_RAMP)
SURFACE, INK, INK_SECONDARY, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'

STYLE = {
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'text.color': INK,
    'figure.titlesize': 12,
    'figure.titleweight': 'bold',
    'axes.titlecolor': INK,
    'axes.titlesize': 10.5,
    'axes.titlelocation': 'left',
    'axes.labelcolor': INK_SECONDARY,
    'axes.labelsize': 9.5,
    'axes.edgecolor': AXIS,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'axes.axisbelow': True,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'grid.linestyle': '-',
    'xtick.color': AXIS,
    'ytick.color': AXIS,
    'xtick.labelcolor': INK_SECONDARY,
    'ytick.labelcolor': INK_SECONDARY,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.frameon': False,
    'legend.fontsize': 9,
    'lines.linewidth': 1.6,
    'lines.solid_capstyle': 'round',
    'lines.solid_joinstyle': 'round',
}


# ----------------------------------------------------------------------------------------------------------------------
# Finding and reading the metrics files.

def find_metrics_dirs(metrics_dir=None):
    """Return the metrics folders to read, the most authoritative first.

    `metrics_dir` (a path or a list of paths) is used as is. Otherwise every `metrics/` folder holding JSON files is
    collected, in this order:
      1. under /kaggle/working - the runs of the current Kaggle session, also while they are still training;
      2. under /kaggle/input - Datasets made from the output of earlier sessions (read-only);
      3. next to this file - the repo itself, e.g. with results downloaded from Kaggle.
    When several folders hold a file with the same name, `load_runs` keeps the first one.
    """
    if metrics_dir is not None:
        paths = [metrics_dir] if isinstance(metrics_dir, (str, os.PathLike)) else metrics_dir
        return [Path(path) for path in paths]

    found = (_metrics_dirs_under(KAGGLE_WORKING, max_depth=3) + _metrics_dirs_under(KAGGLE_INPUT, max_depth=6)
             + _metrics_dirs_under(Path(__file__).resolve().parent, max_depth=1))
    unique = {}
    for path in found:
        unique.setdefault(path.resolve(), path)
    return list(unique.values())


def _metrics_dirs_under(root, max_depth):
    """`metrics/` folders holding at least one JSON file, at most `max_depth` levels below `root`."""
    found = []
    if not root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        if path.name == 'metrics' and any(_is_metrics_file(name) for name in filenames):
            found.append(path)
        depth = len(path.relative_to(root).parts)
        dirnames[:] = [] if depth >= max_depth else sorted(name for name in dirnames if not name.startswith('.'))
    return found


def _is_metrics_file(filename):
    # Names starting with '.' are the temporary files of `utils.save_metrics`.
    return filename.endswith('.json') and not filename.startswith('.')


def load_runs(metrics_dir=None, task=None):
    """Read the metrics files; returns {run name: metrics}, the run name being the file name without `.json`.

    Each metrics dict gets a `_path` entry. Files that cannot be parsed or were not written by the training scripts are
    skipped (with a warning for the former). `task` keeps only the runs of 'sentiment', 'paraphrase' or 'sonnet'.
    """
    runs = {}
    for directory in find_metrics_dirs(metrics_dir):
        if not directory.is_dir():
            warnings.warn(f'{directory} is not a folder')
            continue
        for path in sorted(directory.iterdir()):
            if not _is_metrics_file(path.name) or path.stem in runs:  # an earlier folder already has this run
                continue
            try:
                metrics = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError) as error:
                warnings.warn(f'Skipping {path}: {error}')
                continue
            if isinstance(metrics, dict) and metrics.get('task') in TASKS:
                metrics['_path'] = str(path)
                runs[path.stem] = metrics
    if task is not None:
        runs = {name: metrics for name, metrics in runs.items() if metrics['task'] == task}
    return runs


def list_runs(metrics_dir=None):
    """Show which folders are read and one row per run: progress, saved epoch and its dev score."""
    directories = find_metrics_dirs(metrics_dir)
    if not directories:
        print('No metrics folder found: train with one of the scripts first, or pass metrics_dir=...')
        return
    print('Metrics folders (for a file name found in several of them, the first one is used):')
    for directory in directories:
        print(f'  {directory}')

    runs = load_runs(directories)
    if not runs:
        print('No metrics file in these folders yet.')
        return
    rows = []
    for name, metrics in sorted(runs.items(), key=lambda item: (TASKS.index(item[1]['task']), item[0])):
        config = metrics.get('config', {})
        key, score_name = ('dev_chrf', 'dev chrF') if metrics['task'] == 'sonnet' else ('dev_acc', 'dev acc')
        rows.append({
            'run': name,
            'task': metrics['task'],
            'lr': config.get('lr'),
            'epochs done': _progress(metrics),
            'saved epoch': metrics.get('best_epoch'),
            'score': score_name,
            'value': _at_saved_epoch(metrics).get(key),
            'dev predictions': '-' if metrics['task'] == 'sonnet' else _tested(metrics),
            'file': metrics['_path'],
        })
    _show(pd.DataFrame(rows).set_index('run').round({'value': 3}))


def _select(runs, names):
    """Keep the runs whose name contains one of `names` (a string or a list); None keeps all of them."""
    if names is None:
        return runs
    names = [names] if isinstance(names, str) else list(names)
    return {run: metrics for run, metrics in runs.items() if any(name in run for name in names)}


def _progress(metrics):
    return f"{len(metrics.get('history') or [])}/{metrics.get('config', {}).get('epochs', '?')}"


def _tested(metrics):
    return 'yes' if metrics.get('best_dev') else 'not yet'


def _at_saved_epoch(metrics):
    """The history entry of the epoch whose checkpoint was saved (empty if none yet)."""
    best = metrics.get('best_epoch')
    return next((entry for entry in metrics.get('history') or [] if entry['epoch'] == best), {})


def _number(value):
    return float('-inf') if value is None else value


# ----------------------------------------------------------------------------------------------------------------------
# Output helpers: display in a notebook, print (or only save) elsewhere.

def _in_notebook():
    """True inside a Jupyter kernel (which can display figures and HTML), False in a terminal."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return type(get_ipython()).__name__ == 'ZMQInteractiveShell'


def _show(obj):
    if _in_notebook():
        from IPython.display import display
        display(obj)
    elif not isinstance(obj, plt.Figure):
        print(obj)


def _heading(text):
    if _in_notebook():
        from IPython.display import Markdown, display
        display(Markdown(f'**{text}**'))
    else:
        print(f'\n{text}')


def _suptitle(fig, title):
    """Figure title, wrapped to the width of the figure."""
    fig.suptitle(textwrap.fill(title, max(30, int(fig.get_figwidth() * 8.5))))


def _finish(fig, save_dir, filename):
    """Save `fig` as `save_dir/filename` (when `save_dir` is given), display it, then close it."""
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / filename, dpi=200, bbox_inches='tight')
    _show(fig)
    plt.close(fig)


def _run_colors(all_runs, selected):
    """Series color per run, by sorted name over all runs of the task, so that selecting runs does not repaint any."""
    pool = all_runs if len(all_runs) <= len(SERIES_COLORS) else selected
    if len(pool) > len(SERIES_COLORS):
        raise ValueError(f'{len(pool)} runs to draw but only {len(SERIES_COLORS)} series colors: select some with runs=')
    return {name: SERIES_COLORS[i] for i, name in enumerate(sorted(pool))}


def _run_label(metrics):
    config = metrics.get('config', {})
    return f"{config.get('model_size', '?')} · lr {config.get('lr', float('nan')):g} · {config.get('epochs', '?')} epochs"


# ----------------------------------------------------------------------------------------------------------------------
# Learning curves.

def _plot_metric(ax, metrics, key, color, dots):
    """One run's `key` per epoch (`dots`: a dot on every epoch); the saved epoch gets a larger ringed dot."""
    history = metrics.get('history') or []
    epochs = [entry['epoch'] for entry in history]
    values = [entry.get(key, np.nan) for entry in history]
    ax.plot(epochs, values, color=color, marker='o' if dots else None, markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.2)
    best = metrics.get('best_epoch')
    if best in epochs:
        ax.plot(best, values[epochs.index(best)], linestyle='none', marker='o', markersize=10, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3)


def _plot_curve_grid(rows, panels, colors, labels, criterion, title, save_dir, filename):
    """One row of `panels` ((history key, panel title) pairs) per (row label, {run: metrics}) of `rows`."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(rows), len(panels), figsize=(4.3 * len(panels), 3.2 * len(rows) + 0.7),
                                 squeeze=False, layout='constrained')
        for row_axes, (row_label, runs) in zip(axes, rows):
            # A dot on every epoch only while the row is short enough for the dots not to crowd the lines.
            dots = max(len(metrics.get('history') or []) for metrics in runs.values()) <= 15
            for ax, (key, panel_title) in zip(row_axes, panels):
                for name, metrics in runs.items():
                    _plot_metric(ax, metrics, key, colors[name], dots)
                ax.set_title(f'{row_label} · {panel_title}' if row_label else panel_title)
                ax.set_xlabel('epoch')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # A legend for two series or more (one series is named by the title); the dot key is always needed.
        series = {labels[name]: colors[name] for _, runs in rows for name in runs}
        handles = [Line2D([], [], color=color, label=label) for label, color in series.items()] if len(series) > 1 else []
        handles.append(Line2D([], [], linestyle='none', marker='o', markersize=10, color=INK_SECONDARY,
                              markeredgecolor=SURFACE, markeredgewidth=1.8, label=f'saved checkpoint (best {criterion})'))
        fig.legend(handles=handles, loc='outside lower center', ncols=len(handles))
        _suptitle(fig, title if len(series) > 1 else f'{title} ({next(iter(series))})')
        _finish(fig, save_dir, filename)


def _summary(runs, columns, extra=None):
    """One row per run: progress, saved epoch and the `columns` ({history key: header}) at that epoch."""
    rows = []
    for name, metrics in runs.items():
        saved = _at_saved_epoch(metrics)
        row = {'run': name, 'epochs done': _progress(metrics), 'saved epoch': metrics.get('best_epoch')}
        row.update({header: saved.get(key) for key, header in columns.items()})
        row.update(extra(metrics) if extra else {})
        rows.append(row)
    return pd.DataFrame(rows).set_index('run').round({header: 3 for header in columns.values()})


def plot_sentiment_curves(metrics_dir=None, save_dir=None):
    """Train loss, dev accuracy and dev macro-F1 per epoch: one row per dataset, one line per fine-tuning mode."""
    runs = load_runs(metrics_dir, task='sentiment')
    rows = [(DATASET_NAMES[dataset], _sentiment_runs(runs, dataset)) for dataset in SENTIMENT_CLASSES]
    rows = [(label, dataset_runs) for label, dataset_runs in rows if dataset_runs]
    if not rows:
        print('No sentiment run found (visualize.list_runs() shows the folders that were searched).')
        return
    colors = {name: MODE_COLORS.get(metrics.get('mode'), SERIES_COLORS[len(MODE_COLORS)]) for name, metrics in runs.items()}
    labels = {name: metrics.get('mode', name) for name, metrics in runs.items()}
    panels = [('train_loss', 'train loss'), ('dev_acc', 'dev accuracy'), ('dev_f1', 'dev macro-F1')]
    _plot_curve_grid(rows, panels, colors, labels, 'dev accuracy', 'Sentiment classification: learning curves',
                     save_dir, 'sentiment_curves.png')

    _heading('Values at the saved epoch')
    ordered = {name: metrics for _, dataset_runs in rows for name, metrics in dataset_runs.items()}
    _show(_summary(ordered, {'train_loss': 'train loss', 'train_acc': 'train acc', 'dev_acc': 'dev acc',
                             'dev_f1': 'dev macro-F1'},
                   lambda metrics: {'lr': metrics.get('config', {}).get('lr'), 'dev predictions': _tested(metrics)}))


def _sentiment_runs(runs, dataset):
    """The runs on `dataset`, in the order of SENTIMENT_MODES (unknown modes last)."""
    def order(item):
        mode = item[1].get('mode')
        return SENTIMENT_MODES.index(mode) if mode in SENTIMENT_MODES else len(SENTIMENT_MODES), item[0]

    return dict(sorted(((name, m) for name, m in runs.items() if m.get('dataset') == dataset), key=order))


def plot_paraphrase_curves(metrics_dir=None, runs=None, save_dir=None):
    """Train loss, dev accuracy and dev macro-F1 per epoch, one line per run (`runs`: names or parts of names)."""
    all_runs = load_runs(metrics_dir, task='paraphrase')
    selected = _select(all_runs, runs)
    if not selected:
        print('No paraphrase run found (visualize.list_runs() shows the folders that were searched).')
        return
    labels = {name: _run_label(metrics) for name, metrics in selected.items()}
    panels = [('train_loss', 'train loss'), ('dev_acc', 'dev accuracy'), ('dev_f1', 'dev macro-F1')]
    _plot_curve_grid([(None, selected)], panels, _run_colors(all_runs, selected), labels, 'dev accuracy',
                     'Paraphrase detection (Quora): learning curves', save_dir, 'paraphrase_curves.png')

    _heading('Values at the saved epoch')
    _show(_summary(selected, {'train_loss': 'train loss', 'dev_acc': 'dev acc', 'dev_f1': 'dev macro-F1'},
                   lambda metrics: {'dev predictions': _tested(metrics)}))


def plot_sonnet_curves(metrics_dir=None, runs=None, save_dir=None):
    """Train loss and dev chrF per epoch, one line per run (`runs`: names or parts of names)."""
    all_runs = load_runs(metrics_dir, task='sonnet')
    selected = _select(all_runs, runs)
    if not selected:
        print('No sonnet run found (visualize.list_runs() shows the folders that were searched).')
        return
    labels = {name: _run_label(metrics) for name, metrics in selected.items()}
    _plot_curve_grid([(None, selected)], [('train_loss', 'train loss'), ('dev_chrf', 'dev chrF')],
                     _run_colors(all_runs, selected), labels, 'dev chrF', 'Sonnet generation: learning curves',
                     save_dir, 'sonnet_curves.png')

    def extra(metrics):
        config, history = metrics.get('config', {}), metrics.get('history') or []
        last_chrf = history[-1].get('dev_chrf') if history else None
        return {'last dev chrF': None if last_chrf is None else round(last_chrf, 3),
                'temperature': config.get('temperature'), 'top_p': config.get('top_p')}

    _heading('Values at the saved epoch')
    _show(_summary(selected, {'train_loss': 'train loss', 'dev_chrf': 'dev chrF'}, extra))


# ----------------------------------------------------------------------------------------------------------------------
# Confusion matrices and per-class scores of the saved checkpoint (dev predictions written by test()).

def plot_confusion(ax, y_true, y_pred, class_names, title):
    """Row-normalized confusion matrix: each true class sums to 1, so the diagonal is the per-class recall."""
    k = len(class_names)
    counts = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    totals = counts.sum(axis=1, keepdims=True)
    rates = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)

    # The surface-colored cell edges are the 2px gap between touching fills.
    mesh = ax.pcolormesh(rates, cmap=CONFUSION_CMAP, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=1.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ticks = np.arange(k) + 0.5
    ax.set_xticks(ticks, class_names, rotation=30 if k > 2 else 0, ha='right' if k > 2 else 'center')
    ax.set_yticks(ticks, class_names)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title)
    for i in range(k):
        for j in range(k):
            text, color = '0', INK_SECONDARY  # empty cells stay quiet
            if counts[i, j]:
                text, color = f'{rates[i, j]:.2f}\n({counts[i, j]})', _text_color_on(CONFUSION_CMAP(rates[i, j]))
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=9 if k <= 3 else 8, color=color)
    return mesh


def _text_color_on(fill):
    """Ink or white, whichever contrasts more with `fill` (text set inside a colored cell)."""
    def luminance(color):
        rgb = np.array(to_rgb(color))
        rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    fill_luminance = luminance(fill)
    return 'white' if 1.05 / (fill_luminance + 0.05) > (fill_luminance + 0.05) / (luminance(INK) + 0.05) else INK


def per_class_report(y_true, y_pred, class_names, positive=None):
    """Precision / recall / F1 / support of every class, then the macro and support-weighted averages.

    Both averages run over the classes found in `y_true` or `y_pred`, so 'macro avg' F1 is exactly the `dev f1` that
    the training scripts print (`f1_score(average='macro')`). `positive` (a label id) marks the positive class of a
    binary task: its row is the binary precision / recall / F1 (`average='binary', pos_label=positive`).
    """
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rows = {}
    for i, name in enumerate(class_names):
        rows[f'{i}: {name}' + (' (positive class)' if i == positive else '')] = [precision[i], recall[i], f1[i],
                                                                                    support[i]]
    for average in ('macro', 'weighted'):
        scores = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
        rows[f'{average} avg'] = [*scores[:3], support.sum()]
    report = pd.DataFrame.from_dict(rows, orient='index', columns=['precision', 'recall', 'f1', 'support'])
    return report.astype({'support': int}).round(3)


def _plot_confusions(entries, class_names, title, save_dir, filename):
    """One confusion matrix per (subplot title, metrics) of `entries`, side by side on a shared color scale."""
    side = max(2.7, 0.65 * len(class_names) + 0.4)  # inches per matrix, wide enough for its 3-line title
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(entries), figsize=(len(entries) * (side + 1.4) + 0.9, side + 1.9),
                                 squeeze=False, layout='constrained')
        for ax, (subtitle, metrics) in zip(axes[0], entries):
            dev = metrics['best_dev']
            mesh = plot_confusion(ax, dev['y_true'], dev['y_pred'], class_names,
                                  f"{subtitle}\ndev acc {dev['acc']:.3f} · macro-F1 {dev['f1']:.3f}\n"
                                  f"saved epoch {metrics.get('best_epoch')}")
        colorbar = fig.colorbar(mesh, ax=axes[0].tolist(), shrink=0.8, ticks=[0, 0.25, 0.5, 0.75, 1])
        colorbar.set_label('share of the true class (count in brackets)')
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(length=0)
        _suptitle(fig, title)
        _finish(fig, save_dir, filename)


def _per_class_table(entries, class_names, positive=None):
    """`per_class_report` of every (title, metrics) of `entries`, side by side under their titles."""
    return pd.concat({subtitle: per_class_report(metrics['best_dev']['y_true'], metrics['best_dev']['y_pred'],
                                                 class_names, positive)
                      for subtitle, metrics in entries}, axis=1)


def _tested_entries(runs, title_of):
    """(title, metrics) of the runs that have dev predictions; reports the others."""
    entries = []
    for name, metrics in runs.items():
        if metrics.get('best_dev'):
            entries.append((title_of(name, metrics), metrics))
        else:
            print(f'{name}: no dev predictions yet (test() writes them after the last epoch).')
    return entries


def show_sentiment_results(metrics_dir=None, save_dir=None):
    """Confusion matrix and per-class precision / recall / F1 of the saved checkpoint of every sentiment run (dev)."""
    runs = load_runs(metrics_dir, task='sentiment')
    if not runs:
        print('No sentiment run found (visualize.list_runs() shows the folders that were searched).')
        return
    for dataset, class_names in SENTIMENT_CLASSES.items():
        entries = _tested_entries(_sentiment_runs(runs, dataset), lambda name, metrics: metrics.get('mode', name))
        if entries:
            name = DATASET_NAMES[dataset]
            _plot_confusions(entries, class_names, f'{name}: confusion matrix of the saved checkpoint (dev)',
                             save_dir, f'sentiment_confusion_{dataset}.png')
            _heading(f'{name}: per-class precision / recall / F1 of the saved checkpoint (dev)')
            _show(_per_class_table(entries, class_names))


def show_paraphrase_results(metrics_dir=None, runs=None, save_dir=None):
    """Confusion matrix and precision / recall / F1 of the 'paraphrase' class of the saved checkpoint (dev)."""
    selected = _select(load_runs(metrics_dir, task='paraphrase'), runs)
    if not selected:
        print('No paraphrase run found (visualize.list_runs() shows the folders that were searched).')
        return
    entries = _tested_entries(selected, lambda name, metrics: _run_label(metrics))
    if entries:
        _plot_confusions(entries, PARAPHRASE_CLASSES, 'Paraphrase: confusion matrix of the saved checkpoint (dev)',
                         save_dir, 'paraphrase_confusion.png')
        _heading('Precision / recall / F1 of the saved checkpoint (dev): the "positive class" row is the binary score '
                 'of "paraphrase" (pos_label=1), "macro avg" is the dev f1 printed by the script')
        _show(_per_class_table(entries, PARAPHRASE_CLASSES, positive=1))


# ----------------------------------------------------------------------------------------------------------------------
# Sonnet samples.

def show_sonnet_samples(metrics_dir=None, run=None, sonnet_ids=None, epochs=None):
    """Prompt, generated continuation and gold continuation of dev sonnets at a few epochs of one sonnet run.

    run: run name or part of it; when several runs match, the one with the best dev chrF is shown.
    sonnet_ids: ids of the dev sonnets (their 0-based position in the dev file), 'all', or None for the first one.
    epochs: the epochs to show; None shows the first, middle (index len // 2), best and last epoch.
    """
    runs = {name: metrics for name, metrics in _select(load_runs(metrics_dir, task='sonnet'), run).items()
            if metrics.get('generations')}
    if not runs:
        print('No sonnet run with dev generations found (visualize.list_runs() shows the folders that were searched).')
        return
    name = max(runs, key=lambda run_name: _number(runs[run_name].get('best_dev_chrf')))
    if len(runs) > 1:
        print(f'{len(runs)} sonnet runs match: showing {name} (best dev chrF), pick another one with run=...')
    metrics = runs[name]

    picks = _sample_epochs(metrics, epochs)
    dev_chrf = {entry['epoch']: entry.get('dev_chrf') for entry in metrics.get('history') or []}
    dev = metrics.get('dev') or []
    if sonnet_ids is None:
        sonnet_ids = [entry['id'] for entry in dev[:1]]
    elif sonnet_ids == 'all':
        sonnet_ids = [entry['id'] for entry in dev]
    elif isinstance(sonnet_ids, int):
        sonnet_ids = [sonnet_ids]

    for entry in dev:
        if entry['id'] not in sonnet_ids:
            continue
        samples = []
        for epoch, roles in picks:
            text = next((g['text'] for g in metrics['generations'][str(epoch)] if g['id'] == entry['id']), '')
            caption = f'epoch {epoch}' + (f" ({', '.join(roles)})" if roles else '')
            if dev_chrf.get(epoch) is not None:
                caption += f' · dev chrF of the epoch {dev_chrf[epoch]:.2f}'
            samples.append((caption, _continuation(text, entry['prompt'])))
        config = metrics.get('config', {})
        title = (f"{name} · dev sonnet {entry['id']} · temperature {config.get('temperature')}, "
                 f"top_p {config.get('top_p')}")
        prompt, gold = entry['prompt'], _continuation(entry['gold'], entry['prompt'])
        if _in_notebook():
            from IPython.display import HTML, display
            display(HTML(_sonnet_html(title, prompt, gold, samples)))
        else:
            print(_sonnet_text(title, prompt, gold, samples))


def _sample_epochs(metrics, epochs=None):
    """[(epoch, roles)]: the first, middle, best and last epoch with generations, or the given `epochs`."""
    available = sorted(int(epoch) for epoch in metrics.get('generations', {}))
    if not available:
        return []
    roles = {}
    for role, epoch in (('first', available[0]), ('middle', available[len(available) // 2]),
                        ('best', metrics.get('best_epoch')), ('last', available[-1])):
        if epoch in available:
            roles.setdefault(epoch, []).append(role)
    if epochs is None:
        return [(epoch, roles[epoch]) for epoch in sorted(roles)]
    epochs = [epochs] if isinstance(epochs, int) else epochs
    missing = [epoch for epoch in epochs if epoch not in available]
    if missing:
        print(f'No generations for epochs {missing} (available: {available[0]}..{available[-1]}).')
    return [(epoch, roles.get(epoch, [])) for epoch in epochs if epoch in available]


def _continuation(text, prompt):
    """`text` without the prompt lines it starts with (the dev generations and gold sonnets both begin with them)."""
    if text.startswith(prompt):
        return text[len(prompt):].strip('\n')
    # The decoded text may not reproduce the prompt character for character: drop as many non-empty lines instead.
    remaining = len([line for line in prompt.splitlines() if line.strip()])
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if remaining == 0:
            return '\n'.join(lines[i:]).strip('\n')
        remaining -= bool(line.strip())
    return ''


def _sonnet_html(title, prompt, gold, samples):
    """A 3-column table: the prompt and the gold continuation span the rows, one row per sampled epoch."""
    cell = 'border: 1px solid rgba(127, 127, 127, 0.35); padding: 6px 10px; vertical-align: top; text-align: left'
    verse = ('white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; '
             'font-size: 12px; line-height: 1.45')
    caption = 'font-size: 11px; opacity: 0.7; margin-bottom: 4px'
    rows = []
    for i, (label, text) in enumerate(samples):
        cells = [f'<td style="{cell}"><div style="{caption}">{escape(label)}</div>'
                 f'<div style="{verse}">{escape(text) or "<i>(nothing generated)</i>"}</div></td>']
        if i == 0:
            span = f'rowspan="{len(samples)}"'
            cells.insert(0, f'<td {span} style="{cell}"><div style="{verse}">{escape(prompt)}</div></td>')
            cells.append(f'<td {span} style="{cell}"><div style="{verse}">{escape(gold)}</div></td>')
        rows.append(f'<tr>{"".join(cells)}</tr>')
    head = ''.join(f'<th style="{cell}; width: 33%">{header}</th>'
                   for header in ('Prompt (the 3 given lines)', 'Generated continuation', 'Gold continuation'))
    return (f'<div style="font-weight: 600; margin: 14px 0 6px">{escape(title)}</div>'
            f'<table style="border-collapse: collapse; width: 100%; table-layout: fixed">'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def _sonnet_text(title, prompt, gold, samples):
    parts = [f'=== {title} ===', '--- Prompt (the 3 given lines) ---', prompt, '--- Gold continuation ---', gold]
    for label, text in samples:
        parts += [f'--- Generated continuation, {label} ---', text]
    return '\n'.join(parts) + '\n'
