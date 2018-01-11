drop table g_ratio_ty_lm_2017 ; 
drop table g_ratio_ly_tm_2017 ; 

create table g_ratio_ty_lm_2017
as (
select a.store_nbr, a.item_nbr, sales_ln_m5/sales_ln_m4 as ratio_ty_lm
from 
(
	sel store_nbr, item_nbr, sum((ln(sales+1))) as sales_ln_m5
	from pzhang.g_expand e, 
	pzhang.g_calendar c
	where c.fiscalyear = 2017
	and  e.salesdate = c.salesdate 
	 and fiscalperiod = 5 
	group by 1,2
) a inner join 
(
	sel store_nbr, item_nbr, sum((ln(sales+1))) as sales_ln_m4
	from pzhang.g_expand e, 
	pzhang.g_calendar c
	where c.fiscalyear = 2017
	and  e.salesdate = c.salesdate 
	 and fiscalperiod = 4
	group by 1,2
	having sales_ln_m4 >0 
) b
on a.store_nbr = b.store_nbr
and a.item_nbr = b.item_nbr 
) with data 
primary index( item_nbr ) 
;


create table g_ratio_ly_tm_2017
as (
	select a.store_nbr, a.item_nbr, sales_ln_m6/sales_ln_m5 as ratio_ly_tm
	from 
	(
		sel store_nbr, item_nbr, sum((ln(sales+1))) as sales_ln_m6
		from pzhang.g_expand e, 
		pzhang.g_calendar c
		where c.fiscalyear = 2016
		and  e.salesdate = c.salesdate 
		 and fiscalperiod = 6
		group by 1,2
	) a inner join 
	(
		sel store_nbr, item_nbr, sum((ln(sales+1))) as sales_ln_m5
		from pzhang.g_expand e, 
		pzhang.g_calendar c
		where c.fiscalyear = 2016
		and  e.salesdate = c.salesdate 
		 and fiscalperiod = 5
		group by 1,2
		having sales_ln_m5 >0 
	) b
	on a.store_nbr = b.store_nbr
	and a.item_nbr = b.item_nbr 
) with data 
primary index( item_nbr ) 
;


