def get_x_y(item, kdom):
    # Data from views (first is the last)
    vs = views(item)[::-1]
    vs = [v for v in vs if dom(v['event_info']) is not None]
    iids = [ v['event_info'] for v in vs ]
    dids = list(map(dom,iids))
    cids = list(map(lambda x : item_data[x]['category_id'], iids))
    sh = search_huerfana(item)
    if bool(sh):
        print(sh)
        sh = sh[0][0].split(' ')[:3]
        sh = sh + ['','',''][:3-len(sh)]
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
