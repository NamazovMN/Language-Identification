import torch.cuda

from vocabulary import Vocabulary
from utilities import *
from training import Train
from inference import Inference
from statistics import Statistics

def __main__() -> None:
    """
    Main function of the project which performs all steps
    :return: None
    """
    processed_dir = 'datasets'
    check_dir(processed_dir)
    input_dir = 'input_data'
    check_dir(input_dir)
    params_dir = 'parameters'
    check_dir(params_dir)
    project_parameters = get_parameters()
    save_project_parameters(params_dir, project_parameters)
    data_structure = f"structure_{project_parameters['data_structure']}"
    ds_path = os.path.join(processed_dir, data_structure)
    check_dir(ds_path)
    use_words = project_parameters['use_words']
    exp_num = project_parameters['experiment_num']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_process = Preprocess(
            use_words=use_words,
            window_size=project_parameters['window_size'],
            window_shift=project_parameters['window_shift'],
            remove_html=project_parameters['html_tags'],
            remove_numbers=project_parameters['numbers'],
            remove_punctuation=project_parameters['punctuation'],
            lower_case=project_parameters['lower']
        )
    dataset = collect_dataset(
            load_path='papluca/language-identification',
            data_process=data_process,
            dataset_dir=ds_path
        )

    vocabulary = Vocabulary(source_dir=ds_path, use_words=use_words)

    statistics = Statistics(
        exp_num=exp_num,
        dataset_dir=ds_path,
        use_words=use_words
    )
    if project_parameters['train']:
        hp = {
            'data_structure': data_structure,
            'emb_dim': project_parameters['embedding_dim'],
            'hid_dim': project_parameters['hidden_dim'],
            'num_classes': len(vocabulary.labels),
            'bidirectional': project_parameters['bidirectional'],
            'dropout': project_parameters['lstm_dropout'],
            'num_layers': project_parameters['num_layers'],
            'max_length': project_parameters['max_length'],
            'learning_rate': project_parameters['learning_rate'],
            'batch_size': project_parameters['batch_size'],
            'weight_decay': project_parameters['weight_decay']
        }
        trainer = Train(
            preprocess=data_process,
            vocabulary=vocabulary,
            hyperparams=hp,
            exp_num=exp_num,
            device=device,
            resume_training=project_parameters['resume_training'],
            load_best=project_parameters['load_best'],
            choice=project_parameters['load_choice'],
            epoch_choice=project_parameters['epoch_choice']
        )
        trainer.training(dataset, project_parameters['epochs'])
        statistics.plot_results()

    if project_parameters['infer']:
        inference = Inference(
            experiment_num=exp_num,
            device=device,
            input_dir=input_dir,
            choice_prompt=project_parameters['from_file'],
            load_best=project_parameters['load_best'],
            choice=project_parameters['choice'],
            epoch_choice=project_parameters['epoch_choice'],
        )
        inference.infer_process()



if __name__ == '__main__':
    __main__()

