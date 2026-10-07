"""Customer usage forecasting from the exported, evaluated method."""
import numpy as np
from .customer_service import _get_customer
from scripts.export_models import forecast_series

def forecast(customer_id):
    bundle,history,_,_,metrics,_=_get_customer(customer_id)
    series=history[history.customer_id==customer_id].sort_values("week")
    actual=series.usage_minutes.to_numpy(float)
    values=forecast_series(actual,bundle["forecast_method"])
    spread=float(bundle["forecast_spread"])
    current=float(np.mean(actual[-4:])); end=float(values[-1]); change=(end-current)/current*100 if current>0 else 0
    points=[{"week":int(series.week.max()+i+1),"usage_minutes":float(v),"lower":float(max(0,v-spread)),"upper":float(v+spread)} for i,v in enumerate(values)]
    observed=[{"week":int(w),"usage_minutes":float(v)} for w,v in zip(series.week,actual)]
    return {"customer_id":customer_id,"method":bundle["forecast_method"],"history":observed,"forecast":points,"current_weekly_usage_minutes":current,"week_4_minutes":end,"expected_change_pct":change,"metrics":metrics["forecasting"]}
