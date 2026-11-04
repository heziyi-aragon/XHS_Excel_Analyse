"""Source-traceable creator pricing, without network access or invented market data.

Industry observations are supplied by the caller and retain their provenance.
Internal peers are an explicitly synthetic/sample reference, never live market
evidence. Dates are compared only against an explicit pricing_as_of anchor.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import math
import re
from typing import Any


_CENT = Decimal('0.01')
_FORMS = {'image': 'price_image', 'video': 'price_video'}
_MIN_SAMPLES = 3


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).strip().replace(',', ''))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if number.is_finite() else None


def _money(value: Decimal | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value.quantize(_CENT, rounding=ROUND_HALF_UP))
    except (InvalidOperation, OverflowError) as error:
        raise ValueError('Currency amount is outside the supported finite precision') from error
    if not math.isfinite(result):
        raise ValueError('Currency amount must remain finite when serialized')
    return result


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ''


def _label(value: Any) -> str:
    return _text(value).casefold()


def _iso_date(value: Any, field: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError(f'{field} must be an ISO YYYY-MM-DD date')
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f'{field} must be a valid ISO date') from error


def _identity(record: dict, talent: dict) -> tuple[str, str]:
    return (_text(record.get('xhs_id') or talent.get('xhs_id')),
            _text(record.get('nickname') or talent.get('nickname')))


def _same_creator(creator_id: str, own_id: str, own_name: str) -> bool:
    if not creator_id:
        return False
    aliases = {own_id, 'id:' + own_id} if own_id else set()
    if own_name:
        aliases.update({own_name, 'nickname:' + own_name})
    return creator_id in aliases


def _quantile(values: list[Decimal], fraction: Decimal) -> Decimal:
    """Inclusive, linearly interpolated quantile; no floating-point arithmetic."""
    ordered = sorted(values)
    position = Decimal(len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(samples: list[dict], kind: str, match: dict, anchor: str | None,
             excluded_future: list[str], duplicates: list[str]) -> dict:
    values = [item['_price'] for item in samples]
    retained = list(samples)
    outliers: list[str] = []
    if len(values) >= _MIN_SAMPLES:
        q1, q3 = _quantile(values, Decimal('.25')), _quantile(values, Decimal('.75'))
        iqr = q3 - q1
        lower_fence, upper_fence = q1 - Decimal('1.5') * iqr, q3 + Decimal('1.5') * iqr
        retained = [item for item in samples if lower_fence <= item['_price'] <= upper_fence]
        outliers = [item['id'] for item in samples if not lower_fence <= item['_price'] <= upper_fence]
    adequate = len(retained) >= _MIN_SAMPLES
    prices = [item['_price'] for item in retained]
    stats = ({'median': _quantile(prices, Decimal('.5')),
              'q1': _quantile(prices, Decimal('.25')),
              'q3': _quantile(prices, Decimal('.75'))} if adequate else
             {'median': None, 'q1': None, 'q3': None})
    stats['iqr'] = stats['q3'] - stats['q1'] if adequate else None
    has_fixture = any(item['synthetic'] for item in samples)
    effective_kind = 'industry_fixture' if kind == 'industry_reference' and has_fixture else kind
    return {'kind': effective_kind, 'sample_count': len(retained),
            'matched_count_before_outliers': len(samples), 'adequate_samples': adequate,
            'minimum_independent_samples': _MIN_SAMPLES,
            'match_rule': match, 'pricing_as_of': anchor,
            'as_of_status': 'explicit_anchor' if anchor else 'as_of_not_provided',
            'real_market_data': kind == 'industry_reference' and not has_fixture,
            'source_verification': 'provided_not_independently_verified',
            **{key: _money(value) for key, value in stats.items()},
            'samples': [{key: value for key, value in item.items() if key != '_price'}
                        for item in retained],
            'excluded_future_ids': sorted(set(excluded_future)),
            'duplicate_ids': sorted(set(duplicates)), 'outlier_ids': outliers,
            '_stats': stats}


def _external_samples(config: dict, match: dict, own_id: str, own_name: str,
                      anchor_date: date | None) -> tuple[list[dict], list[str], list[str]]:
    observations = config.get('industry_benchmarks', [])
    if not isinstance(observations, list):
        raise ValueError('industry_benchmarks must be a list')
    validated, by_id, duplicates, future = [], {}, [], []
    for index, item in enumerate(observations):
        prefix = f'industry_benchmarks[{index}]'
        if not isinstance(item, dict):
            raise ValueError(f'{prefix} must be an observation mapping')
        if isinstance(item.get('id'), bool) or not isinstance(item.get('id'), (str, int)) or not _text(item['id']):
            raise ValueError(f'{prefix}.id must be a nonempty identifier')
        for field in ('content_type', 'tier', 'source'):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f'{prefix}.{field} must be nonempty text')
        if not isinstance(item.get('format'), str) or item['format'] not in _FORMS:
            raise ValueError(f'{prefix}.format must be image or video')
        price = _decimal(item.get('price'))
        if price is None or price <= 0 or _money(price) <= 0:
            raise ValueError(f'{prefix}.price must be finite, positive currency, not bool')
        observed_date = _iso_date(item.get('as_of'), prefix + '.as_of')
        if 'synthetic' in item and type(item['synthetic']) is not bool:
            raise ValueError(f'{prefix}.synthetic must be bool when provided')
        creator = item.get('creator_id')
        if creator is not None and (isinstance(creator, bool) or not isinstance(creator, (str, int)) or not _text(creator)):
            raise ValueError(f'{prefix}.creator_id must be a nonempty identifier when provided')
        clean = {'id': _text(item['id']), 'content_type': item['content_type'].strip(),
                 'tier': item['tier'].strip(), 'format': item['format'],
                 'price': _money(price), '_price': price.quantize(_CENT, rounding=ROUND_HALF_UP),
                 'source': item['source'].strip(), 'as_of': observed_date.isoformat(),
                 'creator_id': _text(creator) or None, 'synthetic': item.get('synthetic', False)}
        if clean['id'] in by_id:
            if clean != by_id[clean['id']]:
                raise ValueError(f'Conflicting duplicate industry observation id: {clean["id"]}')
            duplicates.append(clean['id'])
            continue
        by_id[clean['id']] = clean
        validated.append(clean)
    matched = []
    for item in validated:
        if any(_label(item[field]) != _label(match[field]) for field in ('content_type', 'tier', 'format')):
            continue
        if _same_creator(item['creator_id'] or '', own_id, own_name):
            continue
        if anchor_date is not None and date.fromisoformat(item['as_of']) > anchor_date:
            future.append(item['id'])
            continue
        matched.append(item)
    # Several observations for the same creator are not independent peers. Keep
    # the latest eligible observation, with a deterministic id tie-breaker.
    matched.sort(key=lambda item: (item['as_of'], item['id']), reverse=True)
    independent, creators = [], set()
    for item in matched:
        identity = item['creator_id']
        if identity and identity in creators:
            duplicates.append(item['id'])
            continue
        if identity:
            creators.add(identity)
        independent.append(item)
    independent.sort(key=lambda item: item['id'])
    return independent, future, duplicates


def _peer_samples(state: dict, match: dict, own_id: str, own_name: str) -> tuple[list[dict], list[str]]:
    if not match['content_type'] or not match['tier']:
        return [], []
    peers = state.get('talents') or []
    if not isinstance(peers, list):
        raise ValueError('state.talents must be a list for internal peer pricing')
    source = _text(state.get('talent_path')) or 'state.talents (provided sample)'
    result, duplicates, seen = [], [], set()
    for row in peers:
        if not isinstance(row, dict):
            continue
        talent = row.get('talent') if isinstance(row.get('talent'), dict) else row
        if any(_label(talent.get(field)) != _label(match[field]) for field in ('content_type', 'tier')):
            continue
        identity, nickname = _identity(row, talent)
        if (own_id and identity and own_id == identity) or (
                (not own_id or not identity) and own_name and nickname == own_name):
            continue
        key = 'id:' + identity if identity else 'nickname:' + nickname if nickname else ''
        if not key:
            continue
        if key in seen:
            duplicates.append(key)
            continue
        seen.add(key)
        price = _decimal(talent.get(_FORMS[match['format']]))
        if price is None or price <= 0 or _money(price) <= 0:
            continue
        result.append({'id': key, 'content_type': _text(talent.get('content_type')),
                       'tier': _text(talent.get('tier')), 'format': match['format'],
                       'price': _money(price), '_price': price.quantize(_CENT, rounding=ROUND_HALF_UP),
                       'source': source, 'as_of': None, 'date_status': 'not_recorded',
                       'creator_id': identity or None, 'synthetic': True})
    return result, duplicates


def price_for_creator(record: dict, talent: dict, form: str, config: dict | None,
                      state: dict | None) -> dict:
    """Price one creator/form using documented observations and platform bounds.

    Invalid configuration raises ValueError.  Missing platform price returns
    status=unavailable; a benchmark never invents the absent platform anchor.
    """
    if not isinstance(record, dict) or not isinstance(talent, dict):
        raise ValueError('record and talent must be mappings')
    if config is not None and not isinstance(config, dict):
        raise ValueError('pricing config must be a mapping')
    if state is not None and not isinstance(state, dict):
        raise ValueError('pricing state must be a mapping')
    config, state = config or {}, state or {}
    if not isinstance(form, str) or form not in _FORMS:
        raise ValueError('Pricing format must be image or video')
    anchor_value = config.get('pricing_as_of')
    if anchor_value is None:
        anchor_value = state.get('pricing_as_of')
    anchor_date = _iso_date(anchor_value, 'pricing_as_of') if anchor_value is not None else None
    anchor = anchor_date.isoformat() if anchor_date else None
    own_id, own_name = _identity(record, talent)
    match = {'content_type': _text(talent.get('content_type')),
             'tier': _text(talent.get('tier')), 'format': form,
             'exclude_creator_id': own_id or own_name or None,
             'independence': 'unique_observation_id_and_creator_id_when_available'}
    samples, future, duplicates = _external_samples(config, match, own_id, own_name, anchor_date)
    external = _summary(samples, 'industry_reference', match, anchor, future, duplicates)
    attempts = [external]
    benchmark = external if external['adequate_samples'] else None
    if benchmark is None:
        peers, peer_duplicates = _peer_samples(state, match, own_id, own_name)
        peer = _summary(peers, 'sample_peer_reference', match, anchor, [], peer_duplicates)
        attempts.append(peer)
        if peer['adequate_samples']:
            benchmark = peer
    if benchmark is None:
        benchmark = _summary([], 'not_provided', match, anchor, future, duplicates)
        benchmark['real_market_data'] = False
    stats = benchmark['_stats']
    public_benchmark = {key: value for key, value in benchmark.items() if key != '_stats'}
    public_benchmark['attempts'] = [{key: value for key, value in attempt.items() if key != '_stats'}
                                    for attempt in attempts]
    rationale = []
    if anchor is None:
        rationale.append('未提供pricing_as_of，不使用宿主日期，也不声称已排除未来观察。')
    if benchmark['kind'] == 'sample_peer_reference':
        rationale.append('使用模拟输入中的同类、同层、同形式内部同行平台报价参考，不能当作现实行业行情。')
    elif benchmark['kind'] == 'industry_fixture':
        rationale.append('匹配观察包含synthetic样本，仅作行业基准机制演示，不代表现实市场价格。')
    elif benchmark['kind'] == 'industry_reference':
        rationale.append('使用调用方提供并保留来源日期的同行观察，工具未独立核验来源真实性。')
    else:
        rationale.append('少于3个独立可比观察，未形成可靠行业/同行基准，回退平台参考区间。')
    reference = _decimal(talent.get(_FORMS[form]))
    if reference is None or reference <= 0 or _money(reference) <= 0:
        return {'reference': None, 'low': None, 'suggested': None, 'high': None,
                'status': 'unavailable', 'benchmark': public_benchmark,
                'rationale': rationale + ['该形式缺少有效平台参考价，行业/同行观察不替代缺失的报价锚点。'],
                'conflict': False, 'manual_override_outside_benchmark': False,
                'exception': {'outside_platform_corridor': False, 'reason': None, 'requires_approval': False},
                'industry_benchmark_status': benchmark['kind']}
    reference = reference.quantize(_CENT, rounding=ROUND_HALF_UP)
    platform_low = (reference * Decimal('.8')).quantize(_CENT, rounding=ROUND_HALF_UP)
    platform_high = (reference * Decimal('1.2')).quantize(_CENT, rounding=ROUND_HALF_UP)
    low, high, suggested, conflict = platform_low, platform_high, reference, False
    if benchmark['adequate_samples']:
        intersection_low, intersection_high = max(platform_low, stats['q1']), min(platform_high, stats['q3'])
        if intersection_low <= intersection_high:
            low, high = intersection_low, intersection_high
            suggested = min(high, max(low, stats['median']))
            rationale.append('基准中位数确定建议价，报价区间使用基准Q1–Q3与平台±20%走廊的交集。')
        else:
            conflict = True
            suggested = min(platform_high, max(platform_low, stats['median']))
            rationale.append('基准四分位区间与平台±20%走廊无交集；建议取最近平台边界并保留冲突，交最终人工审核。')
    manual_override = False
    if 'price_multiplier' in config:
        multiplier = _decimal(config['price_multiplier'])
        if multiplier is None or multiplier <= 0:
            raise ValueError('price_multiplier must be finite and positive')
        suggested = (reference * multiplier).quantize(_CENT, rounding=ROUND_HALF_UP)
        if suggested <= 0:
            raise ValueError('price_multiplier rounds the proposed price to zero')
        manual_override = suggested < low or suggested > high
        rationale.append('显式price_multiplier优先：建议价按平台参考价乘本轮倍率计算。')
    outside = suggested < platform_low or suggested > platform_high
    reason = config.get('exception_reason')
    if outside and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError('Pricing outside platform +/-20% requires exception_reason')
    if manual_override:
        low, high = min(low, suggested), max(high, suggested)
        rationale.append('显式建议价超出基准交集，显示区间扩展至包含该建议价；平台走廊以外的变动须单独审批。')
    if outside:
        rationale.append('例外建议价已纳入区间上界/下界；例外原因保留，最终人工批准前不视为成交。')
    return {'reference': _money(reference), 'low': _money(low),
            'suggested': _money(suggested), 'high': _money(high),
            'status': benchmark['kind'] if benchmark['adequate_samples'] else 'platform_fallback',
            'benchmark': public_benchmark, 'rationale': rationale, 'conflict': conflict,
            'manual_override_outside_benchmark': manual_override,
            'platform_corridor': {'low': _money(platform_low), 'high': _money(platform_high)},
            'exception': {'outside_platform_corridor': outside,
                          'reason': reason.strip() if isinstance(reason, str) and reason.strip() else None,
                          'requires_approval': outside},
            'industry_benchmark_status': benchmark['kind']}
