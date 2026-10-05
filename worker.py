"""Executa cada implementação BPAD em processo isolado."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "bpad"))
from pathlib import Path
import numpy as np

METHODS = {
 'likelihood': ('baseline.boehmer','LikelihoodPlusAnomalyDetector'),
 'naive': ('baseline.bezerra','NaiveAnomalyDetector'),
 'sampling': ('baseline.bezerra','SamplingAnomalyDetector'),
 'dae': ('baseline.dae','DAE'),
 'binetv3': ('baseline.binet.binet','BINetv3'),
 'binetv2': ('baseline.binet.binet','BINetv2'),
 'gama': ('baseline.GAMA.gama','GAMA'),
 'vae': ('baseline.VAE.vae','VAE'),
 'lae': ('baseline.LAE.lae','LAE'),
 'gae': ('baseline.GAE.gae','GAE'),
 'grasped': ('baseline.GRASPED.grasped','GRASPED'),
 'leverage': ('baseline.leverage','Leverage'),
 'w2vlof': ('baseline.Sylvio','W2VLOF'),
 'vaeocsvm': ('baseline.VAEOCSVM.vaeOCSVM','VAEOCSVM'),
}

def main():
    dataset_name, method, destination, epochs = sys.argv[1:]
    from importlib import import_module
    from utils.dataset import Dataset
    module_name, class_name = METHODS[method]
    cls = getattr(import_module(module_name), class_name)
    dataset = Dataset(dataset_name, beta=0.005, label_percent=0)
    kwargs = {'n_epochs': int(epochs)} if method in ('gama', 'vae', 'lae', 'gae', 'grasped', 'vaeocsvm') else {}
    detector = cls(**kwargs)
    fit_kwargs = {'epochs': int(epochs), 'batch_size': 64} if method in ('dae','binetv2','binetv3') else {}
    detector.fit(dataset, **fit_kwargs)
    scores, _, _ = detector.detect(dataset)
    values = np.asarray(scores, dtype=float).reshape(-1)
    Path(destination).write_text(json.dumps(values.tolist()))

if __name__ == '__main__':
    main()
