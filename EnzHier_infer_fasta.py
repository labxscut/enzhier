import argparse
import os
from CLEAN.utils import *
from CLEAN.infer import infer_maxsep
from CLEAN.dataloader import *
from CLEAN.model import *
from CLEAN.utils import *
from CLEAN.infer import *
from CLEAN.distance_map import get_dist_map

def eval_parse():
    # only argument passed is the fasta file name to infer
    # located in ./data/[args.fasta_data].fasta
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--fasta_data', type=str)
    args = parser.parse_args()
    return args

def infer_maxsep2(train_data, test_data, report_metrics = False, 
                 pretrained=True, model_name=None, gmm = None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(512, 128, device, dtype)
    
    checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
            
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    ensure_dirs("./results")
    out_filename = "results/" +  test_data
    write_max_sep_choices(eval_df, out_filename, gmm=gmm)
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
        pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
        true_label, all_label = get_true_labels('./data/' + test_data)
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print("############ EC calling results using maximum separation ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} ')
        print('-' * 75)

def main():
    args = eval_parse()
    train_data = 'split100'
    test_data = 'inputs/' + args.fasta_data 
    # converting fasta to dummy csv file, will delete after inference
    # esm embedding are taken care of
    prepare_infer_fasta(test_data) 
    # inferred results is in
    # results/[args.fasta_data].csv
    infer_maxsep2(train_data, test_data, report_metrics=False, pretrained=False, 
                  gmm = './data/pretrained/gmm_ensumble.pkl', model_name='trained model')
    # removing dummy csv file
    os.remove("data/"+ test_data +'.csv')
    

if __name__ == '__main__':
    main()
