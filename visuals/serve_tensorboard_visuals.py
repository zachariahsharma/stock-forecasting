"""Display recorded metrics in native TensorBoard; no training is run here."""
from pathlib import Path
import sys,types,json,time,signal
sys.modules['tensorboard.compat.notf']=types.ModuleType('tensorboard.compat.notf')
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard import program

def main():
    ROOT=Path(__file__).resolve().parent
    DATA=ROOT/'results'
    LOGS=ROOT/'tensorboard-recorded-metrics'
    import shutil
    if LOGS.exists():shutil.rmtree(LOGS)
    LOGS.mkdir(exist_ok=True)
    writers={}
    def record(run,tag,step,value):
        if run not in writers:writers[run]=EventFileWriter(str(LOGS/run))
        writers[run].add_event(Event(wall_time=time.time(),step=step,summary=Summary(value=[Summary.Value(tag=tag,simple_value=float(value))])))
    
    for name,tag in [('v1_mlp_raw','V1_MLP/loss_MSE'),('v2_mlp_scaled','V2_Normalized_MLP/loss_MSE'),('v3_gru_leaked','Optional_leakage_demo/loss_MSE'),('v4_gru_corrected','V4_GRU/loss_MSE'),('v5_cnn_bigru','V5_CNN_BiGRU/loss_MSE'),('v6_autoformer','V6_Autoformer/loss_NLL')]:
        r=json.loads((DATA/(name+'.json')).read_text());suffix='nll' if name=='v6_autoformer' else 'mse'
        for row in r['history']:
            record('New_demo_Train',tag,row['epoch'],row['train_'+suffix])
            record('New_demo_Validation',tag,row['epoch'],row['val_'+suffix])
    
    r=json.loads((DATA/'v4_gru_corrected.json').read_text())
    for i,(a,p) in enumerate(zip(r['test_actual'][-90:],r['test_prediction'][-90:])):
        record('New_demo_Actual','V4_GRU/held_out_price_USD',i,a)
        record('New_demo_Forecast','V4_GRU/held_out_price_USD',i,p)
    
    r=json.loads((DATA/'archived-backtest.json').read_text())
    for row in r['series']:record('Original_2024_Unvalidated_Backtest','V3_Archived_Backtest/portfolio_USD',row['step'],row['portfolio_value'])
    r=json.loads((DATA/'archived-2024-loss.json').read_text())
    for tag,run in [('Loss/train','Original_2024_Train'),('Loss/Test','Original_2024_Logged_Test')]:
        for row in r['series'][tag]:record(run,'Original_2024/loss',row['step'],row['value'])
    for w in writers.values():w.flush();w.close()
    tb=program.TensorBoard()
    tb.configure(argv=[None,'--logdir',str(LOGS),'--port','16006','--host','127.0.0.1','--load_fast','false','--reload_interval','0'])
    print('Native TensorBoard:',tb.launch(),flush=True)
    signal.pause()

if __name__ == "__main__":
    main()
