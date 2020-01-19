데이터를 이어 붙이기 : merge, join, concatenate
데이터를 분리(분리크기선택, 데이터 표본 위치, 동일한 키의 변수값 기준) : cut, qcut,groupby
groupby()함수의 반환 객체 - GroupBy(그룹 연산을 위한 모든 필요한 정보를 가지고 있음)

Groupby객체에 컬럼이름, 컬럼이름이 담긴 배열로 색인이 가능 : 반환 객체타입(SeriesGroupBy, DataFrameGroupBy)
groupby()로 분리된 데이터들의 개수를 확인 : Groupby 객체.size()

분리 -> 함수적용(agg(), apply()) -> 병합

행인덱스 (또는 기준열)을 제외한 나머지 모든 열을 행으로 이어붙여서 새로운 데이터 프레임을 생성(wide구조 - long구조) : stack
동일한 컬럼값을 가지는 로우들을 열로 이어 붙여서 새로운 데이터 프레임을 생성(long 구조 -> wide구조) : unstack

행으로 사용할 키, 열로 사용할 키를 지정해서 정렬된 데이터를 반환하는 새로운 데이터 프레임을 생성 함수 : pivot_table()

교차테이블 생성 - crosstab()
