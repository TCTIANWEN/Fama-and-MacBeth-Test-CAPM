import pandas as pd
import numpy as np
from sklearn import datasets, linear_model


def readin():
    mkt = pd.read_csv('mkt.csv')
    msf = pd.read_csv('msf.csv')
    a = np.array(msf['ret'])
    for i in range(0, len(a)):
        a[i] = float(a[i][0:-1]) / 100
    msf['ret'] = a
    return mkt, msf


def dispose_data(msf):
    permnos = np.unique(msf['permno'])
    t_info = {
        'permno': [],
        'bgndate': [],
        'enddate': [],
        'n_samp': [],
    }
    sf_info = pd.DataFrame(t_info)
    for i in range(0, len(permnos)):
        tdata = msf[msf['permno'] == permnos[i]]
        tbgndate = np.amin(tdata['date'])
        tenddate = np.amax(tdata['date'])
        tn_sample = len(tdata)
        if tbgndate<'2005-02-01' and tenddate>'2016-12-30':
            sf_info = sf_info.append({'permno': permnos[i], 'bgndate': tbgndate, 'enddate': tenddate, 'n_samp': tn_sample}, ignore_index=True)

    return sf_info


def cal_betas(mkt, msf, sf_info):
    rsfs = msf[(msf['date'] > '2005-01-01') & (msf['date'] < '2017-01-01')]
    rkts = mkt[(mkt['date'] > '2005-01-01') & (mkt['date'] < '2017-01-01')]
    new = pd.merge(rsfs, rkts, on='date')
    new.sort_values(by=['permno', 'date'], inplace=True)
    new = new.reset_index(drop=True)
    yalldata = {
        'permno': [],
        'date': [],
        'mkt': [],
        'ret_1': [],
        'ret': [],
        'alpa': [],
        'beta': [],
        'stdvare': [],
    }
    alldata = pd.DataFrame(yalldata)
    for i in range(0, len(sf_info)):
        tpermno = sf_info['permno'][i]
        bngindex = new[new['permno'] == tpermno].index.tolist()[0]
        endindex = new[new['permno'] == tpermno].index.tolist()[-1]

        for j in range(bngindex + 119, endindex):
            ys = np.array(new['ret'][j-119:j+1])
            fxs = np.array(new['mkt'][j-119:j+1])
            xs = []
            for k in fxs:
                xs.append([k])
            regr = linear_model.LinearRegression()
            regr.fit(xs, ys)
            tbeta = regr.coef_
            talpa = regr.intercept_
            tstdvare = np.std(ys-regr.predict(xs))
            alldata = alldata.append({
                'permno': tpermno,
                'date': new['date'][j],
                'mkt': new['mkt'][j],
                'ret_1': new['ret'][j + 1],
                'ret': new['ret'][j],
                'alpa': talpa,
                'beta': tbeta[0],
                'stdvare': tstdvare,
            }, ignore_index=True)
        print(i)
    dates = alldata['date']
    dates = np.unique(dates)
    return alldata, dates


def do_regressions(alldata, dates):
    yres = {
        'date': [],
        'gamma0': [],
        'gamma1': [],
        'gamma2': [],
        'gamma3': [],
        'stde': [],
    }
    res = pd.DataFrame(yres)
    for i in range(0, len(dates)):
        tdate = dates[i]
        tdata = alldata[alldata['date'] == tdate]
        betas = np.array(tdata['beta'], dtype=float)
        stdes = np.array(tdata['stdvare'], dtype=float)
        ys = np.array(tdata['ret_1'], dtype=float)
        xs = []
        for i in range(0, len(ys)):
            xs.append([betas[i], betas[i]*betas[i], stdes[i]])
        regr = linear_model.LinearRegression()
        regr.fit(xs, ys)
        tbetas = regr.coef_
        talpa = regr.intercept_
        tstdvare = np.std(ys - regr.predict(xs))
        res = res.append({
            'date': str(tdate),
            'gamma0': float(talpa),
            'gamma1': float(tbetas[0]),
            'gamma2': float(tbetas[1]),
            'gamma3': float(tbetas[2]),
            'stde': float(tstdvare),
        }, ignore_index=True)
    return res


if __name__ == '__main__':
    mkt, msf = readin() # 读取数据
    sf_info = dispose_data(msf) # 预处理数据，筛选数据量较为充足的数据，统计各个个股的始终时间
    sf_info.to_csv('sf_info.csv')
    alldata, dates = cal_betas(mkt, msf, sf_info) # 第一步回归，回归结果报告在alldata.csv中
    alldata.to_csv('alldata.csv')
    result = do_regressions(alldata, dates) # 第二部回归，回归结果报告在result.csv中
    result.to_csv('result.csv')
