import json
import itertools
import functools
from multiprocessing import Pool
import tqdm
import pandas as pd
import numpy as np
from collections import Counter,defaultdict
import random
import pickle
import math
import gc
import wget
import gzip
import shutil
import time
from collections import OrderedDict
from datetime import datetime
from fastai import *
from tqdm import tqdm
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import xgboost as xgb

HYPER_ITEM = 5
def set_hyper_item(x):
    global HYPER_ITEM
    HYPER_ITEM = x
def get_hyper_item():
    global HYPER_ITEM
    return HYPER_ITEM

pd.set_option('display.max_rows', 500)

TRAIN_DATA = 'train_dataset.jl'
TEST_DATA = 'test_dataset.jl'
ITEM_DATA = 'item_data.jl'
N_FOLDS = 5
TRAIN_SPLIT_DATA_FOLD = 'train_split_fold_{}.jl'
VALID_SPLIT_DATA_FOLD = 'valid_split_fold_{}.jl'
TRAIN_SPLIT_DATA = TRAIN_SPLIT_DATA_FOLD.format(0)
VALID_SPLIT_DATA = VALID_SPLIT_DATA_FOLD.format(0)

# DOWNLOAD DATA

def extract(gzfn):
    with gzip.open(gzfn, 'rb') as f_in:
        with open(gzfn[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def download_data():
    print('Donwloading test dataset')
    #wget.download('https://meli-data-challenge.s3.amazonaws.com/2020/test_dataset.jl.gz','test_dataset.jl.gz')
    print('Extracting...')
    extract('test_dataset.jl.gz')
    print('Donwloading train dataset')
    #wget.download('https://meli-data-challenge.s3.amazonaws.com/2020/train_dataset.jl.gz','train_dataset.jl.gz')
    print('Extracting...')
    extract('train_dataset.jl.gz')
    print('Donwloading item data')
    #wget.download('https://meli-data-challenge.s3.amazonaws.com/2020/item_data.jl.gz','item_data.jl.gz')
    print('Extracting...')
    extract('item_data.jl.gz')
    print('Donwloading sample submission')
    #wget.download('https://meli-data-challenge.s3.amazonaws.com/2020/sample_submission.csv','sample_submission.csv')

# PARSE DATA STRUCTURES

def calc_item_data():
    item_data={}
    with open(ITEM_DATA) as f:
        for line in f:
            it = json.loads(line)
            item_data[it['item_id']]=it
    return item_data

def ds(fn=TRAIN_DATA):
    with open(fn) as f:
        for line in f:
            yield json.loads(line)

def calc_domain2items():
    domain2items = defaultdict(list)
    for x in item_data.values():
        domain2items[x['domain_id']].append(x['item_id'])
    return domain2items

def clen(gen): return len([None for x in gen])

def tstot(event_timestamp): return datetime.strptime(event_timestamp,'%Y-%m-%dT%H:%M:%S.%f-0400')

# VISUALIZATION

def nth(i,gen=None):
    if gen is None: gen=ds()
    return next(itertools.islice(gen,i,None))

def user_search_list(user_history):
    return [ (x['event_info'], x['event_timestamp']) for x in user_history if x['event_type']=='search' ]

def user_view_list(user_history):
    result = []
    for x in user_history:
        if x['event_type']=='view':
            res = item_data[x['event_info']]
            res['time']=x['event_timestamp']
            result.append(res)
    return result

def views_df(user_history):
    return pd.json_normalize( user_view_list(user_history))

def calc_lens(): return len([None for x in ds()]), len([None for x in ds(TEST_DATA)])

def dom(id): return item_data[id]['domain_id']

def views(e): return list(filter(lambda x: x['event_type']=='view',e['user_history']))
def searchs(e): return list(filter(lambda x: x['event_type']=='search',e['user_history']))

def search_huerfana(e):
    results = []
    for x in e['user_history'][::-1]:
        if x['event_type']=='search':
            results.append((x['event_info'], x['event_timestamp']))
        else:
            return results
    return results

def xpeek(x):
    print('Item bought:')
    display( item_data[x['item_bought']] )
    print('Views:')
    vdf = views_df(x['user_history'])
    if not vdf.empty:
        display( vdf )
        print('Domains viewed frequency:')
        display( Counter(vdf['domain_id']) )
        print('Views titles:')
        display( list(vdf['title']) )
    print('Searches:')
    display( user_search_list(x['user_history']) )

def peek(i): xpeek(nth(i))

# STATS

REDUCERS = []
REDUCERS_WITH_ITEM_BOUGHT = []
def is_viewed_item(item):
    item_viewed = { x['event_info'] for x in item['user_history'] if x['event_type']=='view' }
    return item['item_bought'] in item_viewed
REDUCERS_WITH_ITEM_BOUGHT.append(is_viewed_item)

def is_viewed_domain(item):
    domain_viewed = { item_data[x['event_info']]['domain_id'] for x in item['user_history'] if x['event_type']=='view' }
    return item_data[item['item_bought']]['domain_id'] in domain_viewed
REDUCERS_WITH_ITEM_BOUGHT.append(is_viewed_domain)

def is_viewed_category(item):
    category_viewed = { item_data[x['event_info']]['category_id'] for x in item['user_history'] if x['event_type']=='view' }
    return item_data[item['item_bought']]['category_id'] in category_viewed
REDUCERS_WITH_ITEM_BOUGHT.append(is_viewed_category)

def searches_len(item):
    return len([None for x in item['user_history'] if x['event_type']=='search'])
REDUCERS.append(searches_len)

def view_len(item):
    return len([None for x in item['user_history'] if x['event_type']=='view'])
REDUCERS.append(view_len)

def items_view_distinct_in_guessed_domain(item):
    d = item_data[item['item_bought']]['domain_id']
    iids = {x['event_info'] for x in item['user_history'] if x['event_type']=='view'}
    iids = {id for id in iids if item_data[id]['domain_id']==d}
    return len(iids)
REDUCERS_WITH_ITEM_BOUGHT.append(items_view_distinct_in_guessed_domain)

def items_view_distinct(item):
    return len({x['event_info'] for x in item['user_history'] if x['event_type']=='view'})
REDUCERS.append(items_view_distinct)

def domain_view_distinct(item):
    return len({item_data[x['event_info']]['domain_id'] for x in item['user_history'] if x['event_type']=='view'})
REDUCERS.append(domain_view_distinct)

def domain_view_distinct_nonull(item):
    ds = (item_data[x['event_info']]['domain_id'] for x in item['user_history'] if x['event_type']=='view')
    ds = {d for d in ds if d is not None}
    return len(ds)

def category_view_distinct(item):
    return len({item_data[x['event_info']]['category_id'] for x in item['user_history'] if x['event_type']=='view'})
REDUCERS.append(category_view_distinct)

def item_is_last_viewed(item):
    h = item['user_history']
    iids = [ x['event_info'] for x in h if x['event_type']=='view' ]
    return bool(iids) and iids[-1]==item['item_bought']
REDUCERS_WITH_ITEM_BOUGHT.append(item_is_last_viewed)

def item_is_most_common(item):
    h = item['user_history']
    iids = [ x['event_info'] for x in h if x['event_type']=='view' ]
    return bool(iids) and Counter(iids).most_common()[0][0]==item['item_bought']
REDUCERS_WITH_ITEM_BOUGHT.append(item_is_most_common)

def domain_is_last_viewed(item):
    h = item['user_history']
    iids = [ x['event_info'] for x in h if x['event_type']=='view' ]
    dids = [ item_data[x]['domain_id'] for x in iids ]
    return bool(dids) and dids[-1]==item_data[item['item_bought']]['domain_id']
REDUCERS_WITH_ITEM_BOUGHT.append(domain_is_last_viewed)

def domain_is_most_common(item):
    h = item['user_history']
    iids = [ x['event_info'] for x in h if x['event_type']=='view' ]
    dids = [ item_data[x]['domain_id'] for x in iids ]
    return bool(dids) and Counter(dids).most_common()[0][0]==item_data[item['item_bought']]['domain_id']
REDUCERS_WITH_ITEM_BOUGHT.append(domain_is_most_common)

def calc_df_ds(fn):
    f_reducers = REDUCERS + ([] if fn==TEST_DATA else REDUCERS_WITH_ITEM_BOUGHT)
    series = []
    for f in f_reducers:
        print(f'Applying {f.__name__}...')
        series.append(pd.Series(data=map(f,ds(fn))).rename(f.__name__))
    df_ds = pd.concat(series, axis=1)
    for c in df_ds.columns:
        df_ds[c]=df_ds[c].astype({c: np.int32}, copy=False)
    if fn!=TEST_DATA:
        df_bought = pd.json_normalize(list(map(lambda x : item_data[x['item_bought']],ds(fn))))
        df_bought.columns = ['item_bought_'+c for c in df_bought.columns]
        df_ds = pd.concat([df_ds,df_bought],axis=1)
    return df_ds

def dom_stats(gen, pred, pred_str):
    len_total = 0
    len_cond = 0
    len_isvieweddomain = 0
    len_domlastview = 0
    len_dommostcommon = 0
    for x in gen:
        len_total += 1
        if pred(x):
            len_cond += 1
            if is_viewed_domain(x):
                len_isvieweddomain    += 1
                len_domlastview   += domain_is_last_viewed(x)
                len_dommostcommon += domain_is_most_common(x)
    print(f'P({pred_str}) = {len_cond/len_total}')
    print(f'P(dom_inview | {pred_str}) = {len_isvieweddomain/len_cond}')

    print(f'P(domain_is_last_viewed | {pred_str}) = {len_dommostcommon/len_cond}')
    print(f'P(domain_is_most_common | {pred_str}) = {len_domlastview/len_cond}')

    print(f'P(domain_is_last_viewed | dom_inview ^ {pred_str}) = {len_dommostcommon/len_isvieweddomain}')
    print(f'P(domain_is_most_common | dom_inview ^ {pred_str}) = {len_domlastview/len_isvieweddomain}')

def dom_stats_lu(l,u):
    print(f'Caso {l} <= |doms_view| =< {u}')
    print('Stats on whole train')
    dom_stats(ds(), lambda x: l<=domain_view_distinct_nonull(x)<=u ,'2_dom')
    print('Stats on valid')
    dom_stats(ds(VALID_SPLIT_DATA), lambda x: l<=domain_view_distinct_nonull(x)<=u ,'2_dom')

# GLOBALS

item_data, train_len, test_len, domain2items = [ None for _ in range(4) ]

def update_globals():
    global item_data, df_items, train_len, test_len, df_ds, domain2items
    print('Loading items...')
    item_data = calc_item_data()

    print('Calculating ds len...')
    train_len,test_len = calc_lens()

    print('Calculating domain -> items mapping...')
    domain2items = calc_domain2items()
    with open('objs.pkl', 'wb') as f:
        pickle.dump([item_data, train_len, test_len, domain2items], f)
    print('Succesfully update_globals!')

def restore_globals():
    global item_data, train_len, test_len, domain2items
    with open('objs.pkl', 'rb') as f:
        item_data, train_len, test_len, domain2items = pickle.load(f)
    return item_data, train_len, test_len, domain2items

df_items, df_ds, df_ds_test, df_ds_valid = [ None for _ in range(4) ]

def update_globals_dfs():
    # df_items
    df_items = pd.json_normalize(list(item_data.values()))
    df_items.to_csv('items.csv', index=False)
    del df_items
    gc.collect()
    # df_ds, df_ds_test, df_ds_valid
    for df_name, fn in [('train.csv', TRAIN_DATA), ('test.csv', TEST_DATA), ('valid.csv', VALID_SPLIT_DATA)]:
        print(f'Calculating {df_name}...')
        df_ds = calc_df_ds(fn)
        df_ds.to_csv(df_name, index=False)
        del df_ds
        gc.collect()

def restore_globals_dfs():
    global df_items, df_ds, df_ds_test, df_ds_valid
    df_items = pd.read_csv('items.csv')
    df_ds = pd.read_csv('train.csv')
    df_ds_test = pd.read_csv('test.csv')
    df_ds_valid = pd.read_csv('valid.csv')
    return df_items, df_ds, df_ds_test, df_ds_valid

# SPLIT

def generate_split_idxs(valid_ratio=0.2, seed=42):
    random.seed(seed)
    idxs = list(range(train_len))
    random.shuffle(idxs)
    return set(idxs[:int(valid_ratio*train_len)])

def write_train_and_valid(valid_idxs_set, fn_train=TRAIN_SPLIT_DATA, fn_valid=VALID_SPLIT_DATA):
    with open(TRAIN_DATA) as f_in, open(fn_train, 'w') as f_train, open(fn_valid, 'w') as f_valid:
        for i,line in enumerate(f_in):
            if i in valid_idxs_set:
                f_valid.write(line)
            else:
                f_train.write(line)

def split_valid_train(): write_train_and_valid(generate_split_idxs())

def generate_split_idxs_folds(n_folds=N_FOLDS, seed=42):
    random.seed(seed)
    idxs = list(range(train_len))
    random.shuffle(idxs)
    valid_len = int(train_len/n_folds)
    return [set(idxs[valid_len*i : valid_len*(i+1)]) for i in range(n_folds)]

def split_valid_train_cross(n_folds=N_FOLDS):
    split_valid_idxs = generate_split_idxs_folds(n_folds)
    for i in range(n_folds):
        write_train_and_valid(split_valid_idxs[i],
                              TRAIN_SPLIT_DATA_FOLD.format(i),
                              VALID_SPLIT_DATA_FOLD.format(i))

# SCORING

def withcoefs(relevances):
    coef = np.array( [ 1/math.log(i+2) for i in range(10) ] )
    return (coef*relevances).sum()

def idcg(n):
    perfect_score = np.array([12]+[1 for _ in range(9)])
    return withcoefs(perfect_score)*n

def relevance(pred,target):
    if pred==target: return 12
    if item_data[pred]['domain_id']==item_data[target]['domain_id']: return 1
    return 0

def dcg(preds, item_bought):
    already_12 = False
    relevances = []
    items_before = set()
    for pred in preds:
        if pred in items_before:
            r = 0
        else:
            r = relevance(pred, item_bought)
        items_before.add(pred)
        if r==12:
            if already_12:
                r = 1
            already_12 = True
        relevances.append(r)
    #print(l)
    return withcoefs(np.array(relevances))

# EVALUATION

def read_submit(fn):
    with open(fn) as f:
        for line in f: yield map(int,line.split(','))

def read_submit_list(fn):
    with open(fn) as f:
        return [ list(map(int,line.split(','))) for line in f ]

def get_item_bought(x): return x['item_bought']

def eval_preds(answer_gen, target_gen):
    return sum( dcg(preds, item_bought) for preds, item_bought in zip(answer_gen, target_gen) )

def eval_predictor(predictor, get_gen=None, fn=None):
    if fn is not None:
        assert get_gen is None
        get_gen = lambda : ds(fn)
    else:
        assert get_gen is not None
    data_len = clen(get_gen())
    answer_gen = tqdm(map(predictor, get_gen()), total=data_len)
    target_gen = map(get_item_bought, get_gen())
    return eval_preds(answer_gen, target_gen)/idcg(data_len)

def cross_validate(predictor):
    scores = []
    for i in range(N_FOLDS):
        print(f'Fold {i}')
        if hasattr(predictor,'fit'):
            predictor.fit(TRAIN_SPLIT_DATA_FOLD.format(i))
        scores.append( eval_predictor(predictor, fn=VALID_SPLIT_DATA_FOLD.format(i)) )
    return np.mean(scores), scores

def write_predictor(predictor, solution_fn='solution.csv', fn=TEST_DATA):
    data_len=clen(ds(fn))
    with open(solution_fn,'w') as f:
        for l in tqdm(map(predictor,ds(fn)),total=data_len):
            f.write(','.join(map(str,l))+'\n')

# HEURISTICS

def oraculo_repeat(item): return [get_item_bought(item)]*10

def most_common_heuristic_repeat(e):
    h = e['user_history']
    iids = [ x['event_info'] for x in h if x['event_type']=='view' ]
    if iids:
        p = Counter(iids).most_common()[0][0]
    else:
        p = 1335130
    return [p]*10

def most_common_heuristic_better_cheater(e):
    h = e['user_history']
    iids = [ x['event_info'] for x in h if x['event_type']=='view' ]
    ans = []
    if iids:
        dids = [ item_data[x]['domain_id'] for x in iids ]
        did = item_data[e['item_bought']]['domain_id']
        if did not in dids:
            did = Counter(dids).most_common()[0][0]
        iids = [ x for x in iids if item_data[x]['domain_id']==did ]
        ans = list(zip(*Counter(iids).most_common()))[0]
        ans = list(ans)
        if len(ans)>=10:
            ans = ans[:10]
        else:
            sans = set(ans)
            otherids = list(filter(lambda x : x not in sans,domain2items[did]))
            ans = ans + random.sample(otherids,min(10-len(ans),len(otherids)))
    if len(ans)<10:
        ans = ans + [98853, 379167, 1098739, 1203256, 1379249, 1484614, 756385, 119703, 1197101, 859574][:10-len(ans)]
    return ans

def most_common_heuristic(e):
    h = e['user_history']
    iids = [ x['event_info'] for x in h if x['event_type']=='view' ]
    ans = []
    if iids:
        dids = [ item_data[x]['domain_id'] for x in iids ]
        did = Counter(dids).most_common()[0][0]
        iids = [ x for x in iids if item_data[x]['domain_id']==did ]
        ans = list(zip(*Counter(iids).most_common()))[0]
        ans = list(ans)
        if len(ans)>=10:
            ans = ans[:10]
        else:
            sans = set(ans)
            otherids = list(filter(lambda x : x not in sans,domain2items[did]))
            ans = ans + random.sample(otherids,min(10-len(ans),len(otherids)))
    if len(ans)<10:
        ans = ans + [98853, 379167, 1098739, 1203256, 1379249, 1484614, 756385, 119703, 1197101, 859574][:10-len(ans)]
    return ans

def last_item_heuristic_repeat(e):
    h = e['user_history']
    iids = [ x['event_info'] for x in h if x['event_type']=='view' ]
    if iids:
        p = iids[-1]
    else:
        p = 1335130
    return [p]*10

def train_bought(gen, f_map=lambda x : x):
    pi_no_views = Counter(f_map(x['item_bought']) for x in gen)
    t = sum(pi_no_views.values())
    pi_no_views={k:v/t for k,v in pi_no_views.most_common()}
    return pi_no_views
def calc_iv(pds,pis):
    return sorted( ( (iid, (pi*HYPER_ITEM+1)*pds[dom(iid)]) for iid, pi in pis.items() ), key=lambda x: x[1],reverse=True)


def ext_ans(ans, more_ids):
    for x in more_ids:
        if len(ans)==10: break
        if x not in ans:
            if type(x) is not int:
                print(x)
                assert False
            ans.append(x)
    return ans

def ext_ans_ev(ans, ev): return ext_ans(ans, (i for i,_ in ev))

class CasitosHeuristic:
    def __init__(self, is_item_hyper =  0.5956):
        self.is_item_hyper = is_item_hyper
        self.noviews = [98853, 379167, 1098739, 1203256, 1379249, 1484614, 756385, 119703, 1197101, 859574]
    def fit(self, fn):
        self.chooser = defaultdict(Counter)
        for x in ds(fn):
            i = x['item_bought']
            d = dom(i)
            self.chooser[d][i]+=1
    def __call__(self,item):
        myviews = views(item)
        iids = [v['event_info'] for v in myviews]
        icount = Counter(iids)
        dids = [dom(iid) for iid in iids]
        dcount = Counter(dids)
        answer = []
        if myviews:
            views_len = len(myviews)
            # List[(domain_id, prob)]
            p_dom = [(k,v/views_len) for k,v in dcount.items()]
            p_dom_map = dict(p_dom)
            # List[(item_id, prob)]
            p_id = [(k,v/dcount[dom(k)]) for k,v in icount.items()]
            # List[(value, 'domain' | 'item', id)]
            ev_doms = [(pdom, 'domain', did) for did, pdom in p_dom ]
            ev_items = [ (p_dom_map[dom(id)] * (1+HYPER_ITEM*pid*self.is_item_hyper), 'item', id) for id, pid in p_id ]
            ev = sorted(ev_doms + ev_items, key=lambda x : x[0], reverse=True)
            for _, et, id in ev:
                if len(answer)==10: return answer
                candidatos = self.f_choosebydom(id) if et=='domain' else [id]
                answer = ext_ans(answer, candidatos)
        if len(answer)==10: return answer
        return ext_ans(answer, self.noviews)

    def f_choosebydom(self, d):
        mc = self.chooser[d].most_common()
        if not mc: return []
        return list(list(zip(*mc))[0])

class CombinerHeuristic:
    def __init__(self):
        """ cb_fit + cb_proc, fills:
            - iviews: las views
            - p_dom (Dict[str, float_prob]): probabilidad de que ocurra cada dominio que aparece dado que views!=0
              p_dom[''] es la probabilidad de que no ocurra ninguno de los que aparece.
            - p_items (List[int_id,float_prob]): probabilidad de que ocurra item, dado que ocurre dom(item)
            - i_noviews (List[int_id] sorted by value): items ordenados por valor, dado que views es vacio
            - ev_nodom (OrderedDict[int_id, float_value]): probabilidad de que ocurra cada item dado que
              dom='', osea ninguno de los dominios del view es."""
    def __call__(self,item):
        self.proc(item)
        ans = []
        if self.iviews:
            ev_dom = { i: self.p_dom[dom(i)]*(1+HYPER_ITEM*pi) for i,pi in self.p_items }
            proba_no_dom = self.p_dom['']
            for i in ev_dom:
                for pref in ['MLB', 'MLM']:
                    if i in self.ev_nodom[pref]:
                        ev_dom[i] += proba_no_dom*self.ev_nodom[pref][i]*self.p_pref[pref]
            ev_dom = sorted( ev_dom.items(), key=lambda x : x[1], reverse=True)
            min_ev_dom = ev_dom[-1][1]
            for pref in ['MLB', 'MLM']:
                for iid, value in self.ev_nodom[pref].items():
                    new_value = proba_no_dom*value*self.p_pref[pref]
                    if new_value<min_ev_dom: break
                    ev_dom.append((iid,new_value))
            ans=ext_ans_ev(ans, ev_dom)
        if len(ans)<10: ans=ext_ans(ans,self.i_noviews)
        assert len(ans)==10
        return ans

def precalc_fn(calc_name, fn): return calc_name+'_'+fn[:-2]+'pkl'

def classic_pracalc(fn):
    get_gen_no_views = lambda : filter(lambda x : view_len(x)==0, ds(fn))
    calc_pi_no_views(get_gen_no_views, precalc_fn('pi_no_views',fn))
    calc_boughts_totals(ds(fn), precalc_fn('boughts_totals',fn))
    get_gen_view_domain = lambda : filter(lambda x : not is_viewed_domain(x), ds(fn))
    calc_ev_nodom(get_gen_view_domain, precalc_fn('ev_nodom',fn))

def calc_pi_no_views(get_gen, fn_out):
    pi_no_views = train_bought(get_gen(), lambda x : x)
    pd_no_views = train_bought(get_gen(), dom)
    i_noviews = list(zip(*itertools.islice(calc_iv(pd_no_views, pi_no_views),10)))[0]
    pickle.dump(i_noviews, open(fn_out, "wb"))

def calc_boughts_totals(gen, fn_out):
    boughts = defaultdict(Counter)
    for x in gen:
        i = x['item_bought']
        boughts[dom(i)][i]+=1
    boughts_totals = { d: sum(itemcounter.values()) for d,itemcounter in boughts.items() }
    pickle.dump((boughts, boughts_totals), open(fn_out, "wb"))

def calc_ev_nodom(get_gen, fn_out):
    ev_nodom = {}
    for pref in ['MLB', 'MLM']:
        get_gen_pref = lambda : filter(lambda x : dom(x['item_bought'])[:3]==pref, get_gen())
        pi_no_dom = train_bought(get_gen_pref(), lambda x : x)
        pd_no_dom = train_bought(get_gen_pref(), dom)
        ev_nodom[pref] = OrderedDict(calc_iv(pd_no_dom, pi_no_dom))
    pickle.dump(ev_nodom, open(fn_out, "wb"))

def cb_fit_casitos(self, fn):
    self.i_noviews = pickle.load( open( precalc_fn('pi_no_views', fn), "rb" ) )
    self.boughts, self.boughts_totals = pickle.load( open( precalc_fn('boughts_totals', fn), "rb" ) )
    self.ev_nodom = pickle.load( open( precalc_fn('ev_nodom', fn), "rb" ) )

def calc_pdom_trivial(self, item):
    proba_domain_in_view = 0.5275938818
    self.p_dom = { k: proba_domain_in_view*v/self.views_len for k,v in self.dcount.items() }
    self.p_dom[''] = (1-proba_domain_in_view)

def cb_proc_casitos(self, item):
    self.iviews = [v for v in views(item) if dom(v['event_info']) is not None]
    iids = [v['event_info'] for v in self.iviews]
    icount = Counter(iids)
    dids = [dom(iid) for iid in iids]
    self.dcount = Counter(dids)
    self.views_len = len(self.iviews)
    if self.views_len>0:
        predids = Counter(d[:3] for d in dids)
        self.p_pref = {}
        for pref in ['MLB', 'MLM']:
            self.p_pref[pref] = predids[pref]/len(dids)

        self.calc_pdom(item)
        if len(self.p_dom)!=len(self.dcount)+1:
            print(self.dcount)
            print(self.p_dom)
            assert False
        proba_item_in_view = 0.5956234027
        p_items_inview = { k: proba_item_in_view*v/self.dcount[dom(k)] for k,v in icount.items() }
        self.p_items = {}
        for d in set(dids):
            td = self.boughts_totals.get(d,0)
            if td>0:
                for i, cant in self.boughts[d].items():
                    self.p_items[i] = (1-proba_item_in_view)*cant/td + p_items_inview.get(i,0)
            if td<10:
                for i in domain2items[d][:10-td]:
                    self.p_items[i] = 0
        for i,pi in p_items_inview.items():
            if i not in self.p_items:
                self.p_items[i]=pi
        self.p_items=list(self.p_items.items())


def get_x_y(item, kdom):
    # Data from views (first is the last)
    vs = views(item)[::-1]
    vs = [v for v in vs if dom(v['event_info']) is not None]
    iids = [ v['event_info'] for v in vs ]
    dids = list(map(dom,iids))
    cids = list(map(lambda x : item_data[x]['category_id'], iids))
    sh = search_huerfana(item)
    if bool(sh):
        sh = sh[0][0].split(' ')[:3]
        sh = sh + ['','',''][:3-len(sh)]
    else:
        sh = ['','','']
    if 'item_bought' not in item:
        dbought='NOIMPORTA'
    else:
        dbought = dom(item.get('item_bought'))
    # times referenced to end
    ts = [ tstot(v['event_timestamp']) for v in vs ]
    ts = [(ts[0]-t) for t in ts]
    ts = [ t.total_seconds() for t in ts]
    ts_ratios = [ 0 if ts[-1]==0 else t/ts[-1] for t in ts]
    t_coef = [ 1/math.log(2+t) for t in ts ]

    dcnt = Counter(dids)

    # domain -> [0,1 ..] (from last)
    d2i  = {}
    c = 0
    # domain -> [item indexes]
    d2ids = defaultdict(list)
    for i,d in enumerate(dids):
        d2ids[d].append(i)
        if d not in d2i:
            d2i[d]=c
            c+=1
    # list of domain (d2i inverse)
    i2d=[0]*len(d2i)
    for d,i in d2i.items():
        i2d[i]=d

    y = d2i.get(dbought,kdom)
    most_common_d, most_common_cnt = dcnt.most_common(1)[0]
    most_common_didx = d2i[most_common_d]
    dcnt_ratio = {d: c/most_common_cnt for d,c in dcnt.items()}
    d_tsavg =  {d: np.array([ts_ratios[idx] for idx in d2ids[d]]).mean() for d in d2i}
    d_most_common_cat = {d: Counter(cids[idx] for idx in d2ids[d]).most_common()[0][0] for d in d2i}
    d_cntwithts = {d: sum([t_coef[idx] for idx in d2ids[d]]) for d in d2i}
    most_cntts_valued = d2i[max(d_cntwithts.keys(), key=lambda x: d_cntwithts[x])]

    fill = [math.nan]*(kdom-len(i2d))
    group_columns = i2d + fill
    group_columns += [dcnt[d] for d in i2d] + fill
    group_columns += [dcnt_ratio[d] for d in i2d ] + fill
    group_columns += [d_tsavg[d] for d in i2d ] + fill
    group_columns += [d_most_common_cat[d] for d in i2d ] + fill
    group_columns += [d_cntwithts[d] for d in i2d ] + fill

    return group_columns + [most_common_didx, most_cntts_valued,sh[0],sh[1],sh[2], y], i2d+fill

get_x_rowgroups = ['domain_{}']
get_x_rowgroups.append('dcnt_{}')
get_x_rowgroups.append('dcnt_ratio_{}')
get_x_rowgroups.append('dts_{}')
get_x_rowgroups.append('most_common_cat_{}')
get_x_rowgroups.append('cntwithts_{}')

get_x_rowfixed = ['most_common', 'most_cntts', 'sh0', 'sh1', 'sh2']
def get_x_y_colnames(kdom):
    return [g.format(i) for g in get_x_rowgroups for i in range(kdom)] + get_x_rowfixed + ['domain_bought']

def domain_data(gen, ldom, kdom, is_valid):
    data = []
    for item in gen:
        if ldom<=domain_view_distinct_nonull(item)<=kdom:
            x_y = get_x_y(item,kdom)
            x_y = x_y[0]
            data.append(x_y+[is_valid])
    return data

def domain_df(train_gen, valid_gen, ldom ,kdom, fn_out=None):
    ds = domain_data(train_gen, ldom, kdom, is_valid=False)
    ds_valid = domain_data(valid_gen, ldom, kdom, is_valid=True)
    columns_x=get_x_y_colnames(kdom)
    df = pd.DataFrame(data = ds+ds_valid, columns=columns_x+['is_valid'])
    if fn_out is not None:
        df.to_csv(index=False)
    return df

# Domain predictor

def fastai_tabular(df,kdom):
    train_idx = np.where(~df.is_valid)[0]
    valid_idx = np.where(df.is_valid)[0]
    splits = (list(train_idx),list(valid_idx))

    procs = [Categorify, FillMissing]
    dep_var = 'domain_bought'
    cont,cat = cont_cat_split(df, 1, dep_var=dep_var)
    cat.remove('is_valid')
    print('Continius var:')
    print(cont)
    print('Category var:')
    print(cat)
    to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)
    return to

def to_stats(to, kdom, verbose=True):
    xs,y = to.train.xs,to.train.y
    valid_xs,valid_y = to.valid.xs,to.valid.y
    if verbose:
        print(f'len train: {len(xs)} / Llen valid: {len(valid_xs)}')
    p_domainview = (y!=kdom).mean()
    if verbose:
        print(f'p_domainview: {p_domainview} (on train)')
    p_mostcommon_valid = (valid_xs['most_common']==valid_y).mean()
    p_lastview_valid = (0==valid_y).mean()
    if verbose:
        print(f'mostcommon {p_mostcommon_valid} (valid)')
        print(f'lastview {p_lastview_valid} (valid)')

    n_samples = valid_xs.shape[0]
    n_classes = kdom+1
    correct_proba = np.zeros( (n_samples, n_classes) )
    correct_proba[ range(n_samples), valid_y ]=1

    most_common_proba = np.zeros((n_samples,n_classes))
    most_common_proba[range(n_samples), valid_xs.most_common] = p_mostcommon_valid
    most_common_proba[range(valid_xs.shape[0]),kdom] = 1-p_mostcommon_valid

    lastview_proba = np.zeros((n_samples,n_classes))
    lastview_proba[range(n_samples), 0] = p_lastview_valid
    lastview_proba[range(valid_xs.shape[0]),kdom] = 1-p_lastview_valid

    mse_mostcommon = mean_squared_error(correct_proba,lastview_proba)
    mse_lastview = mean_squared_error(correct_proba,most_common_proba)

    if verbose:
        print(f'mse_mostcommon: {mse_mostcommon}')
        print(f'mse_lastview: {mse_lastview}')
    return min(mse_mostcommon,mse_lastview)

def mse_improve(to, kdom, mse_heuristic, predict_proba, verbose=True):
    valid_xs,valid_y = to.valid.xs,to.valid.y
    n_samples = valid_xs.shape[0]
    n_classes = kdom+1
    correct_proba = np.zeros( (n_samples, n_classes ) )
    correct_proba[ range(n_samples), valid_y ]=1
    valid_proba = predict_proba(valid_xs)
    mse_valid = mean_squared_error(correct_proba,valid_proba)
    print(f'mse_valid: {mse_valid}')
    print(f'mse_improve_ratio: {(mse_heuristic-mse_valid)/mse_heuristic}')
    return mse_valid,(mse_heuristic-mse_valid)/mse_heuristic

class ProbaGeneratorClassifier:
    def __init__(self, to=None, m=None, l=None, kdom=None): self.to, self.m, self.l, self.kdom = to, m, l, kdom
    def predict_proba(self, xs): return self.m.predict_proba(xs)
    def save(self,fn): joblib.dump([self.to, self.m, self.l, self.kdom], fn)
    def load(self,fn): self.to, self.m, self.l, self.kdom=joblib.load(fn)

class ProbaGeneratorRegressors:
    def __init__(self, to=None, ms=None, l=None, kdom=None): self.to, self.ms, self.l, self.kdom = to, ms, l, kdom
    def predict_proba(self, xs): return  np.array([self.ms[c].predict(xs) for c in range(self.kdom+1)]).transpose()
    def save(self,fn): joblib.dump([self.to, self.ms, self.l, self.kdom], fn)
    def load(self,fn): self.to, self.ms, self.l, self.kdom=joblib.load(fn)

def pg_condition(pg, domain_nonull_count): return pg.l<=domain_nonull_count<=pg.kdom

def pg_mse_improve(pg):
    mse_heuristic = to_stats(pg.to, pg.kdom)
    mse_improve(pg.to, pg.kdom, mse_heuristic, pg.predict_proba)

def pg_predict_item(pg, item):
    data,d2i = get_x_y(item,pg.kdom)
    row = pd.DataFrame(data=[data],columns=get_x_y_colnames(pg.kdom))
    to_row = pg.to.new(row)
    to_row.process()
    pdoms = pg.predict_proba(to_row.train.xs)[0]
    return dict(zip(d2i+[''], pdoms))

def pg_ds(pg, gen):
    data = []
    d2is = []
    for item in gen:
        dsc = domain_view_distinct_nonull(item)
        if pg_condition(pg,dsc):
            x_y,d2i = get_x_y(item,pg.kdom)
            data.append(x_y)
            d2is.append(d2i+[''])
    df = pd.DataFrame(data = data, columns=get_x_y_colnames(pg.kdom))
    to_news = pg.to.new(df)
    to_news.process()
    p = pg.predict_proba(to_news.train.xs)
    p_doms = list(map(lambda x : list(zip(x[0],x[1])),zip(d2is,p)))
    assert all([len(p_dom)==pg.kdom+1 for p_dom in p_doms])
    return p_doms

def rfclassifier(to, n_estimators=None, max_samples=None, max_features=None, min_samples_leaf=None, **kwargs):
    xs,y = to.train.xs,to.train.y
    valid_xs,valid_y = to.valid.xs,to.valid.y

    if n_estimators is None: n_estimators=30
    if max_samples is None: max_samples = int(2/3*len(xs))
    if max_features is None: max_features = 0.5
    if min_samples_leaf is None: min_samples_leaf = 5

    m = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,max_samples=max_samples,
                                max_features=max_features,min_samples_leaf=min_samples_leaf,
                                oob_score=True,**kwargs).fit(xs, y)
    print(f'valid accuracy: {(m.predict(valid_xs)==valid_y).mean()}')
    print(f'oob_score :{m.oob_score_}')
    assert kdom+1==m.n_classes_
    return m

def xgboostclassifier(to, kdom,**kwargs):
    xs,y = to.train.xs,to.train.y
    valid_xs,valid_y = to.valid.xs,to.valid.y
    m = xgb.XGBClassifier(verbosity=0,**kwargs).fit(xs, y)
    return m

def rfregressors(to, kdom, n_estimators=None, max_samples=None, max_features=None, min_samples_leaf=None, **kwargs):
    xs,y = to.train.xs,to.train.y
    valid_xs,valid_y = to.valid.xs,to.valid.y

    if n_estimators is None: n_estimators=30
    if max_samples is None: max_samples = int(2/3*len(xs))
    if max_features is None: max_features = 0.5
    if min_samples_leaf is None: min_samples_leaf = 5

    rs = []
    for c in range(kdom+1):
        r = RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,max_samples=max_samples,
                                    max_features=max_features,min_samples_leaf=min_samples_leaf,
                                    oob_score=True,**kwargs).fit(xs, (y==c).astype('float'))
        """
        xgbr = xgb.XGBRegressor(verbosity=0)
        xgbr.fit(xs, (y==c).astype('float'))
        """
        rs.append(r)
    return rs



def predict_pdom(x, tos, ms):
    dc =  len({d for d in (dom(v['event_info']) for v in views(x)) if d is not None})
    if dc<=2:
        bucket = 0
    elif dc<=5:
        bucket = 1
    elif dc<=17:
        bucket = 2
    else:
        assert False
    kdom = LU_SPLITS[bucket][1]
    data,d2i = get_x_y(x,kdom)
    row=pd.DataFrame(data=[data],columns=get_x_y_colnames(kdom))
    to_row = tos[bucket].new(row)
    to_row.process()
    pdoms = ms[bucket].predict_proba(to_row.train.xs)[0]
    return dict(zip(d2i+[''], pdoms))
