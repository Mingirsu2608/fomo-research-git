"""Keep the standalone notebook's embedded implementation aligned with src."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/baseline_notenook.ipynb'


def notebook_cells():
    return json.loads(NOTEBOOK.read_text())['cells']


def test_notebook_has_no_repo_imports_or_absolute_input_paths():
    for cell in notebook_cells():
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('src')
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith('src') for alias in node.names)
        assert '/Users/' not in source
        assert 'sys.path.insert' not in source
        assert 'load_config(' not in source
        assert 'read_text(' not in source or 'features.txt' in source


def test_embedded_implementations_match_source():
    tagged = {tag: ''.join(cell['source']) for cell in notebook_cells()
              for tag in cell.get('metadata', {}).get('tags', [])}
    modules = {
        'inline_features': ('src/features/build.py', {'build_features'}),
        'inline_preprocessing': ('src/models/preprocess.py', set()),
        'inline_training': ('src/models/train.py', {'train_models'}),
    }
    for tag, (source, excluded) in modules.items():
        embedded = {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(tagged[tag]).body if isinstance(node, ast.FunctionDef)}
        for node in ast.parse((ROOT / source).read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name not in excluded:
                assert embedded[node.name] == ast.dump(node, include_attributes=False), node.name

    # Training adapters differ in I/O, but the fitting/tuning loop stays identical.
    def function_tree(source, name):
        return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    source_train = function_tree((ROOT / 'src/models/train.py').read_text(), 'train_models')
    notebook_train = function_tree(tagged['inline_training'], 'train_models')
    source_loop = next(n for n in source_train.body if isinstance(n, ast.For))
    notebook_loop = next(n for n in notebook_train.body if isinstance(n, ast.For))
    assert ast.dump(source_loop) == ast.dump(notebook_loop)
    source_eval = function_tree((ROOT / 'src/evaluation/evaluate.py').read_text(), 'evaluate_models')
    notebook_eval = function_tree(tagged['inline_evaluation'], 'evaluate_models')
    source_loop = next(n for n in source_eval.body if isinstance(n, ast.For))
    notebook_loop = next(n for n in notebook_eval.body if isinstance(n, ast.For))
    # First statement loads a model from disk in src, or gets it from memory here.
    assert [ast.dump(n) for n in source_loop.body[1:]] == [ast.dump(n) for n in notebook_loop.body[1:]]


def test_embedded_config_matches_repository_defaults():
    import yaml
    setup = next(cell for cell in notebook_cells()
                 if 'standalone_setup' in cell.get('metadata', {}).get('tags', []))
    assignments = [node for node in ast.parse(''.join(setup['source'])).body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'config' for t in node.targets)]
    embedded = ast.literal_eval(assignments[0].value)
    defaults = yaml.safe_load((ROOT / 'params.yaml').read_text())
    assert embedded['seed'] == defaults['seed']
    assert embedded['preprocessing'] == defaults['preprocessing']
    for group in ['features', 'model', 'data']:
        for key, value in embedded[group].items():
            assert value == defaults[group][key]
    assert 'tracking' not in embedded
    assert 'output_dir' not in embedded['model']


def test_research_notebook_does_not_export_artifacts_or_use_tracking():
    for cell in notebook_cells():
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        assert 'mlflow' not in source.lower()
        assert 'joblib' not in source.lower()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Attribute):
                assert node.attr not in {'to_csv', 'write_text', 'write_bytes', 'mkdir', 'dump'}
