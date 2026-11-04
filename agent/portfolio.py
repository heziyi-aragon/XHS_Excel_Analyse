"""Step 9: build a budgeted, provisional creator portfolio without changing step 8.

The solver uses Python's standard library and integer cents.  Historical medians
are planning proxies, never a promise of future delivery.  ``config['targets']``
accepts exposure/interactions/cpe/brand_search_lift entries of the form
``{'value': 250000, 'source': 'client confirmed brief, section ...'}``.

The default one-post cap is a development assumption, not a client requirement.
Use max_posts_per_creator/per_creator_max_posts to change it explicitly.  The
sum of image and video posts for one creator is subject to that same cap.
Use per_creator_allowed_formats={'creator-id': ['video']} when a consultant
explicitly restricts a creator's deliverable, without editing source prices.
Structured feedback may contain config_overrides; supported source-grounded
text is translated before solving and unknown changes require clarification.
To revise, pass the old result in state['portfolio_plan'].  Nothing in
the input state, including creator attempt/score/reasons, is ever modified.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
import math
import re
from typing import Any

from agent.portfolio_config import merge_config, paths_of
from agent.portfolio_planning import prepare_portfolio_config
from agent.portfolio_pricing import price_for_creator


_CENT = Decimal('0.01')
_ZERO = Decimal(0)
_TARGET_NAMES = ('exposure', 'interactions', 'cpe', 'brand_search_lift')
_RISK_NAMES = ('data_incomplete', 'missing_data_fields', 'release_status',
               'audit_status', 'audit_issues')
_OVERRIDE_NAMES = frozenset({'price_multiplier', 'exception_reason',
                           'max_posts_per_creator', 'per_creator_max_posts',
                           'per_creator_allowed_formats', 'max_search_nodes',
                           'targets', 'budget', 'industry_benchmarks', 'pricing_as_of'})


class PortfolioPlanningError(ValueError):
    def __init__(self, planning: dict):
        self.planning = planning
        super().__init__('报价修改尚不能执行，请明确以下要求：' +
                         '; '.join(str(item) for item in planning.get('pending') or []))


def _number(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().replace(',', '').replace('，', '')
        if not value or value.casefold() in {'nan', 'none', '-', '—', 'n/a', '无'}:
            return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return result if result.is_finite() else None


def _float(value: Decimal | int | None) -> float | None:
    return None if value is None else float(value)


def _cents(value: Decimal) -> int:
    return int((value.quantize(_CENT, rounding=ROUND_HALF_UP) * 100))


def _money(value: int) -> float:
    return float(Decimal(value) / 100)


def _json_copy(value: Any) -> Any:
    """Copy risk data into JSON-safe output without repairing the original."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None
    if isinstance(value, dict):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_copy(item) for item in value]
    return str(value)


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be an integer >= {0 if allow_zero else 1}')
    return value


def _scaled_number(text: str, unit: str) -> Decimal:
    value = Decimal(text.replace(',', '').replace('，', ''))
    return value * {'万': Decimal(10000), '亿': Decimal(100000000),
                    '千': Decimal(1000), '': Decimal(1)}[unit]


def _targets(state: dict, cfg: dict) -> dict[str, dict]:
    result = {name: {'value': None, 'source': None} for name in _TARGET_NAMES}
    brief = state.get('brief') or {}
    profile = state.get('requirement_profile') or {}
    sources = [('brief.kpi', brief.get('kpi')),
               ('brief.raw.核心KPI', (brief.get('raw') or {}).get('核心KPI')),
               ('requirement_profile.kpi', profile.get('kpi'))]
    patterns = {
        'exposure': r'(?:总曝光|曝光(?:量|数)?)(?:\s*(?:至少|达到|不低于))?\s*[≥>=：:]*\s*([\d,.，]+)\s*([万亿千]?)',
        'interactions': r'(?:互动总量|总互动量|互动量|互动)(?:\s*(?:至少|达到|不低于))?\s*[≥>=：:]*\s*([\d,.，]+)\s*([万亿千]?)',
        'cpe': r'CPE\s*(?:不超过|至多)?\s*[≤<=：:]*\s*[¥￥]?\s*([\d,.，]+)\s*([万亿千]?)',
        'brand_search_lift': r'(?:品牌词搜索指数|品牌搜索指数|品牌词搜索)(?:提升|增长|增幅)?\s*[≥>=：:]*\s*([\d.]+)\s*(%)',
    }
    for source, text in sources:
        if not isinstance(text, str):
            continue
        for name, pattern in patterns.items():
            if result[name]['value'] is not None:
                continue
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = (Decimal(match[1]) / 100 if name == 'brand_search_lift'
                         else _scaled_number(match[1], match[2]))
                if value >= 0:
                    result[name] = {'value': value, 'source': source,
                                    'source_quote': match[0]}
    overrides = cfg.get('targets') or {}
    if not isinstance(overrides, dict):
        raise ValueError('targets must be a mapping with documented sources')
    for name, item in overrides.items():
        if name not in result or not isinstance(item, dict):
            raise ValueError(f'Unsupported or undocumented target: {name}')
        value = _number(item.get('value'))
        source = item.get('source')
        if value is None or value < 0 or not isinstance(source, str) or not source.strip():
            raise ValueError(f'target {name} needs a finite nonnegative value and source')
        result[name] = {'value': value, 'source': source.strip(),
                        'source_quote': str(item.get('source_quote') or '')}
    return result


def _budget_limit(state: dict, cfg: dict) -> tuple[int, str]:
    if 'budget' in cfg:
        item = cfg['budget']
        if not isinstance(item, dict) or not isinstance(item.get('source'), str) or not item['source'].strip():
            raise ValueError('budget override requires value and a nonempty source')
        value, source = _number(item.get('value')), item['source'].strip()
    else:
        brief = state.get('brief') or {}
        value, source = _number(brief.get('total_budget')), 'brief.total_budget'
        if value is None and isinstance(brief.get('total_budget'), str):
            match = re.search(r'[¥￥]?\s*([\d,.，]+)\s*([万亿千]?)', brief['total_budget'])
            if match:
                value = _scaled_number(match[1], match[2])
    if value is None or value < 0:
        raise ValueError('A finite nonnegative total budget is required')
    return _cents(value), source


def _identity(record: dict, talent: dict) -> tuple[str, str | None, str | None]:
    outer, inner = record.get('xhs_id'), talent.get('xhs_id')
    if outer and inner and str(outer).strip() != str(inner).strip():
        raise ValueError('Candidate identity conflicts with nested talent identity')
    identity = str(outer or inner or '').strip()
    nickname = str(record.get('nickname') or talent.get('nickname') or '').strip()
    if not identity and not nickname:
        raise ValueError('Candidate needs xhs_id or nickname')
    return ('id:' + identity if identity else 'nickname:' + nickname,
            identity or None, nickname or None)


def _prepare_candidates(state: dict, cfg: dict, multiplier: Decimal,
                        exception: bool, warnings: list[str]) -> tuple[list[dict], list[dict]]:
    records = state.get('pricing_candidates') or []
    if not isinstance(records, list):
        raise ValueError('pricing_candidates must be a list')
    default_cap = _positive_int(cfg.get('max_posts_per_creator', 1), 'max_posts_per_creator')
    per_creator = cfg.get('per_creator_max_posts') or {}
    if not isinstance(per_creator, dict):
        raise ValueError('per_creator_max_posts must be a mapping')
    for value in per_creator.values():
        _positive_int(value, 'per_creator_max_posts value', allow_zero=True)
    allowed_formats = cfg.get('per_creator_allowed_formats', {})
    if not isinstance(allowed_formats, dict):
        raise ValueError('per_creator_allowed_formats must be a mapping')
    for identity_key, formats in allowed_formats.items():
        if (not isinstance(identity_key, str) or not identity_key.strip()
                or not isinstance(formats, list) or not formats
                or any(not isinstance(form, str) or form not in {'image', 'video'} for form in formats)
                or len(set(formats)) != len(formats)):
            raise ValueError('per_creator_allowed_formats requires nonempty identity keys and unique image/video lists')
    seen, candidates, conditional = set(), [], []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Each pricing candidate must be a mapping')
        if record.get('stance') != 'recommend':
            warnings.append('A non-recommended record was ignored; its upstream stance was preserved.')
            continue
        if record.get('release_status') == 'held_for_review':
            warnings.append('A held_for_review record was ignored; it is not an available candidate.')
            continue
        talent = record.get('talent')
        if not isinstance(talent, dict):
            raise ValueError('pricing_candidates require a nested talent mapping')
        key, identity, nickname = _identity(record, talent)
        if key in seen:
            raise ValueError(f'Duplicate candidate identity: {key}')
        seen.add(key)
        score = _number(record.get('score'))
        if score is None:
            score = _ZERO
            warnings.append(f'{key}: missing/invalid upstream score contributes zero to this objective.')
        metrics = {}
        for name, field in [('exposure', 'exposure_med'), ('read', 'read_med'),
                            ('interactions', 'interact_med')]:
            value = _number(talent.get(field))
            metrics[name] = value if value is not None and value >= 0 else None
        cap = per_creator.get(identity, per_creator.get(key, per_creator.get(nickname, default_cap)))
        requested = allowed_formats.get(identity, allowed_formats.get(key, allowed_formats.get(nickname)))
        entry = {'creator_key': key, 'xhs_id': identity, 'nickname': nickname,
                 'score': score, 'metrics': metrics, 'cap': cap,
                 'main_format': str(talent.get('main_format') or ''),
                 'risk_flags': {name: _json_copy(record[name]) for name in _RISK_NAMES if name in record},
                 'formats': {}}
        for form, field in [('image', 'price_image'), ('video', 'price_video')]:
            if requested is not None and form not in requested:
                continue
            reference = _number(talent.get(field))
            if reference is None or reference <= 0 or _cents(reference) <= 0:
                warnings.append(f'{key}: {field} is unavailable; this format cannot be budgeted.')
                continue
            valuation = price_for_creator(record, talent, form, cfg, state)
            suggested = _cents(_number(valuation['suggested']))
            if suggested <= 0:
                raise ValueError('price_multiplier rounds a suggested price to zero')
            low = _cents(_number(valuation['low']))
            high = _cents(_number(valuation['high']))
            entry['formats'][form] = {'reference': _cents(reference), 'low': low,
                                      'suggested': suggested, 'high': high,
                                      'exception': exception, 'valuation': valuation}
        if not entry['formats']:
            conditional.append({'creator_key': key, 'xhs_id': identity,
                                'nickname': nickname,
                                'reason': ('not_available_for_requested_format'
                                           if requested is not None else 'missing_valid_price'),
                                'requested_formats': list(requested) if requested is not None else ['image', 'video'],
                                'budget_included': False, 'risk_flags': entry['risk_flags']})
        elif cap > 0:
            candidates.append(entry)
        else:
            warnings.append(f'{key}: excluded from this revision by its explicit zero post cap.')
    candidates.sort(key=lambda item: item['creator_key'])
    return candidates, conditional


def _gap(exposure: Decimal, interactions: Decimal, targets: dict) -> Decimal:
    result = _ZERO
    for name, value in [('exposure', exposure), ('interactions', interactions)]:
        target = targets[name]['value']
        if target is not None and target > 0:
            result += max(_ZERO, (target - value) / target)
    return result


def _cpe_gap(cost: int, interactions: Decimal, target: Decimal | None) -> Decimal:
    if target is None:
        return _ZERO
    if interactions <= 0:
        return Decimal('Infinity')
    excess = max(_ZERO, Decimal(cost) / 100 - target * interactions)
    if target == 0:
        return _ZERO if excess == 0 else Decimal('Infinity')
    return excess / (target * interactions)


def _objective(exposure: Decimal, interactions: Decimal, score: Decimal,
               form_preference: Decimal, cost: int, targets: dict,
               missing_exposure: int, missing_interactions: int) -> tuple:
    primary_gap = _gap(exposure, interactions, targets)
    cpe_gap = (_cpe_gap(cost, interactions, targets['cpe']['value'])
               if not missing_interactions else
               (_ZERO if targets['cpe']['value'] is None else Decimal('Infinity')))
    evaluable = (not (targets['exposure']['value'] is not None and missing_exposure)
                 and not ((targets['interactions']['value'] is not None
                           or targets['cpe']['value'] is not None) and missing_interactions))
    all_numeric_targets_met = primary_gap == 0 and cpe_gap == 0 and evaluable
    return (0 if all_numeric_targets_met else 1, primary_gap, cpe_gap,
            -score, -form_preference, cost)


def _options(candidate: dict, available: int, prefer_image: bool):
    """Yield feasible per-creator allocations lazily, including choosing nobody."""
    forms = candidate['formats']
    # For equal effects and quantity, a form with no higher upper price and no
    # worse soft preference dominates the other.  This is a solver reduction,
    # not a new client restriction on content formats.
    if 'image' in forms and 'video' in forms:
        image_preference, video_preference = 1, 1
        if prefer_image or '图文为主' in candidate['main_format']:
            image_preference, video_preference = 1, 0
        elif '视频为主' in candidate['main_format']:
            image_preference, video_preference = 0, 1
        if (forms['image']['high'] <= forms['video']['high']
                and image_preference >= video_preference):
            forms = {'image': forms['image']}
        elif (forms['video']['high'] <= forms['image']['high']
              and video_preference >= image_preference):
            forms = {'video': forms['video']}
    cap = min(candidate['cap'], available // min(item['high'] for item in forms.values()))
    if len(forms) == 1:
        form = next(iter(forms))
        for quantity in range(cap, 0, -1):
            yield (quantity, 0, quantity * forms[form]['high']) if form == 'image' else (0, quantity, quantity * forms[form]['high'])
        yield 0, 0, 0
        return
    for quantity in range(cap, 0, -1):
        image_high, video_high = forms['image']['high'], forms['video']['high']
        delta = image_high - video_high
        minimum, maximum = 0, quantity
        if delta > 0:
            maximum = min(maximum, (available - quantity * video_high) // delta)
        elif delta < 0:
            needed = quantity * video_high - available
            minimum = max(minimum, (needed + (-delta) - 1) // (-delta))
        for images in range(maximum, minimum - 1, -1):
            videos = quantity - images
            yield images, videos, images * image_high + videos * video_high
    yield 0, 0, 0


def _solve(candidates: list[dict], budget: int, targets: dict, max_nodes: int,
           prefer_image: bool) -> dict:
    count = len(candidates)
    # A stack avoids Python's recursion depth limit.  No solver dependency is needed.
    nodes, stopped = 0, False
    best = {'choices': [], 'objective': _objective(_ZERO, _ZERO, _ZERO, _ZERO, 0, targets, 0, 0),
            'exposure': _ZERO, 'interactions': _ZERO}
    suffix_e, suffix_i, suffix_s = [_ZERO] * (count + 1), [_ZERO] * (count + 1), [_ZERO] * (count + 1)
    suffix_p = [0] * (count + 1)
    for index in range(count - 1, -1, -1):
        item = candidates[index]
        suffix_e[index] = suffix_e[index + 1] + (item['metrics']['exposure'] or _ZERO) * item['cap']
        suffix_i[index] = suffix_i[index + 1] + (item['metrics']['interactions'] or _ZERO) * item['cap']
        suffix_s[index] = suffix_s[index + 1] + max(_ZERO, item['score'])
        suffix_p[index] = suffix_p[index + 1] + item['cap']
    # Event: index, choices, cost, exposure, interactions, score, images, posts,
    # matching forms, missing-exposure lines, missing-interaction lines.
    stack = [(0, [], 0, _ZERO, _ZERO, _ZERO, 0, 0, 0, 0, 0)]
    while stack:
        event = stack.pop()
        if isinstance(event, dict):
            # Resume one lazy per-creator allocation, then descend before its sibling.
            try:
                n_image, n_video, added_cost = next(event['iterator'])
            except StopIteration:
                continue
            index, choices, cost, exposure, interactions, score, images, posts, matches, missing_e, missing_i = event['parent']
            candidate = candidates[index]
            quantity = n_image + n_video
            format_text = candidate['main_format']
            matched = n_image if '图文为主' in format_text else (n_video if '视频为主' in format_text else quantity)
            stack.append(event)
            stack.append((index + 1, choices + [(n_image, n_video)], cost + added_cost,
                          exposure + (candidate['metrics']['exposure'] or _ZERO) * quantity,
                          interactions + (candidate['metrics']['interactions'] or _ZERO) * quantity,
                          score + (candidate['score'] if quantity else _ZERO),
                          images + n_image, posts + quantity, matches + matched,
                          missing_e + int(quantity > 0 and candidate['metrics']['exposure'] is None),
                          missing_i + int(quantity > 0 and candidate['metrics']['interactions'] is None)))
            continue
        if nodes >= max_nodes:
            stopped = True
            break
        index, choices, cost, exposure, interactions, score, images, posts, matches, missing_e, missing_i = event
        nodes += 1
        form_preference = (Decimal(images if prefer_image else matches) / posts) if posts else _ZERO
        objective = _objective(exposure, interactions, score, form_preference,
                               cost, targets, missing_e, missing_i)
        if objective < best['objective']:
            best = {'choices': choices, 'objective': objective,
                    'exposure': exposure, 'interactions': interactions}
        if index == count:
            continue
        future_posts = suffix_p[index]
        matched_so_far = images if prefer_image else matches
        optimistic_form = (Decimal(matched_so_far + future_posts) / (posts + future_posts)
                           if posts + future_posts else _ZERO)
        # For CPE the optimistic denominator takes every remaining interaction,
        # but the numerator ignores all future cost.  It is a valid lower bound,
        # even when a costly low-interaction candidate must eventually be dropped.
        optimistic_objective = _objective(exposure + suffix_e[index],
                                          interactions + suffix_i[index],
                                          score + suffix_s[index], optimistic_form,
                                          cost, targets, missing_e, missing_i)
        if optimistic_objective >= best['objective']:
            continue
        candidate = candidates[index]
        stack.append({'parent': event, 'iterator': iter(_options(candidate, budget - cost, prefer_image))})
    best.update(search_nodes=nodes, solver_status='limit_reached' if stopped else 'optimal')
    return best


def _lines(candidates: list[dict], choices: list[tuple[int, int]]) -> list[dict]:
    result = []
    for candidate, allocation in zip(candidates, choices):
        for form, quantity in zip(('image', 'video'), allocation):
            if not quantity:
                continue
            price = candidate['formats'][form]
            result.append({
                'creator_key': candidate['creator_key'], 'xhs_id': candidate['xhs_id'],
                'nickname': candidate['nickname'], 'format': form, 'quantity': quantity,
                'platform_reference_price': _money(price['reference']),
                'unit_price_low': _money(price['low']),
                'unit_price_suggested': _money(price['suggested']),
                'unit_price_high': _money(price['high']),
                'line_cost_low': _money(price['low'] * quantity),
                'line_cost_suggested': _money(price['suggested'] * quantity),
                'line_cost_high': _money(price['high'] * quantity),
                'score': _float(candidate['score']),
                'effect_proxy': {**{name: _float(value * quantity) if value is not None else None
                                    for name, value in candidate['metrics'].items()},
                                 'basis': 'historical_median_times_posts'},
                'risk_flags': deepcopy(candidate['risk_flags']),
                'price_status': 'proposed',
                'requires_final_human_approval': True,
                'price_exception': price['exception'],
                'valuation': deepcopy(price['valuation']),
            })
    return result


def _kpi_checks(lines: list[dict], targets: dict, total_suggested: int, total_high: int) -> dict:
    checks = {}
    for name in ('exposure', 'interactions'):
        missing = sum(line['effect_proxy'][name] is None for line in lines)
        known = sum((_number(line['effect_proxy'][name]) or _ZERO for line in lines), _ZERO)
        target = targets[name]['value']
        estimate = known if not missing else None
        status = ('not_set' if target is None else 'not_evaluable' if missing else
                  'proxy_met' if known >= target else 'proxy_unmet')
        checks[name] = {'target': _float(target), 'estimate': _float(estimate),
                        'known_estimate': _float(known), 'missing_effect_lines': missing,
                        'status': status, 'gap': _float(max(_ZERO, target - known))
                        if target is not None and not missing else None,
                        'source': targets[name]['source'],
                        'source_quote': targets[name].get('source_quote'),
                        'basis': 'sum_of_historical_medians_times_posts',
                        'verified_delivery': False}
    interaction = checks['interactions']['estimate']
    denominator = _number(interaction)
    cpe = Decimal(total_suggested) / 100 / denominator if denominator is not None and denominator > 0 else None
    conservative = Decimal(total_high) / 100 / denominator if denominator is not None and denominator > 0 else None
    target = targets['cpe']['value']
    checks['cpe'] = {'target': _float(target), 'estimate': _float(conservative),
                     'suggested_estimate': _float(cpe),
                     'conservative_estimate': _float(conservative),
                     'status': 'not_set' if target is None else 'not_evaluable' if conservative is None else
                     'proxy_met' if conservative <= target else 'proxy_unmet',
                     'gap': _float(max(_ZERO, conservative - target))
                     if target is not None and conservative is not None else None,
                     'source': targets['cpe']['source'],
                     'source_quote': targets['cpe'].get('source_quote'),
                     'basis': 'portfolio_cost_divided_by_total_interactions; status_uses_upper_cost',
                     'verified_delivery': False}
    target = targets['brand_search_lift']['value']
    checks['brand_search_lift'] = {
        'target': _float(target), 'estimate': None, 'gap': None,
        'status': 'not_set' if target is None else 'not_evaluable',
        'source': targets['brand_search_lift']['source'],
        'source_quote': targets['brand_search_lift'].get('source_quote'),
        'reason': 'No brand search baseline or supported uplift model was supplied.',
        'verified_delivery': False,
    }
    return checks


def build_portfolio_plan(state: dict, config: dict | None = None, *,
                         llm_client: Any = None, feedback: Any = None) -> dict:
    """Return a new JSON-safe step-9 proposal.  Invalid contracts raise ValueError.

    status=ready means a complete budgeted *planning proxy*, pending human review.
    best_effort means one or more supplied targets cannot currently be supported.
    solver_status=optimal certifies this bounded discrete objective only.  A search
    limit always yields limit_reached, even if the incumbent looks satisfactory.
    """
    if not isinstance(state, dict) or (config is not None and not isinstance(config, dict)):
        raise ValueError('state and config must be mappings')
    old_plan = state.get('portfolio_plan') or state.get('portfolio_previous_plan') or {}
    if not isinstance(old_plan, dict):
        raise ValueError('Previous portfolio_plan must be a mapping')
    patch = deepcopy(config or {})
    cfg = merge_config(old_plan.get('configuration') or {}, patch)
    feedback_patch = {}
    if isinstance(feedback, dict) and 'config_overrides' in feedback:
        overrides = feedback['config_overrides']
        if not isinstance(overrides, dict) or set(overrides) - _OVERRIDE_NAMES:
            raise ValueError('feedback.config_overrides contains unsupported keys')
        feedback_patch = deepcopy(overrides)
        cfg = merge_config(cfg, overrides)
    fingerprint = hashlib.sha256(json.dumps(_json_copy({name: state.get(name) for name in
        ('brief', 'requirement_profile', 'client_answers', 'pricing_candidates', 'talents', 'pricing_as_of',
         'active_policy', 'approved_policy', 'audit_context', 'audit_history', 'audit_summary',
         'audited_talents', 'audit_pending', 'supplement_results', 'audit_unverified_checks')}),
        sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    feedback_id = (feedback.get('feedback_id') or feedback.get('review_id') or feedback.get('id')) if isinstance(feedback, dict) else None
    if (feedback_id is not None and old_plan.get('feedback_id') == str(feedback_id)
            and old_plan.get('input_fingerprint') == fingerprint
            and old_plan.get('configuration') == _json_copy(cfg)):
        return deepcopy(old_plan)
    cfg['__planning_context__'] = {'explicit_paths': paths_of(patch),
                                  'user_feedback_patch_paths': paths_of(feedback_patch)}
    cfg, planning = prepare_portfolio_config(state, cfg, llm_client=llm_client, feedback=feedback)
    if planning['status'] == 'needs_clarification':
        raise PortfolioPlanningError(planning)
    multiplier = _number(cfg.get('price_multiplier', 1))
    if multiplier is None or multiplier <= 0:
        raise ValueError('price_multiplier must be finite and positive')
    exception = not Decimal('0.8') <= multiplier <= Decimal('1.2')
    reason = cfg.get('exception_reason')
    if exception and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError('A price outside the default +/-20% corridor needs exception_reason')
    limit, budget_source = _budget_limit(state, cfg)
    targets = _targets(state, cfg)
    max_nodes = _positive_int(cfg.get('max_search_nodes', 100000), 'max_search_nodes')
    warnings: list[str] = []
    candidates, conditional = _prepare_candidates(state, cfg, multiplier, exception, warnings)
    direction = str((state.get('brief') or {}).get('content_direction') or '')
    prefer_image = '图文为主' in direction or '图文种草为主' in direction
    solution = (_solve(candidates, limit, targets, max_nodes, prefer_image) if candidates else
                {'choices': [], 'solver_status': 'not_run', 'search_nodes': 0})
    lines = _lines(candidates, solution['choices'])
    totals = {name: sum((_cents(_number(line['line_cost_' + name])) for line in lines), 0)
              for name in ('low', 'suggested', 'high')}
    if totals['high'] > limit:
        raise RuntimeError('Internal portfolio budget invariant failed')
    checks = _kpi_checks(lines, targets, totals['suggested'], totals['high'])
    configured_checks = [item for item in checks.values() if item['target'] is not None]
    supported = bool(configured_checks) and all(item['status'] == 'proxy_met' for item in configured_checks)
    status = ('needs_pricing' if not candidates and conditional else 'no_candidates' if not candidates else
              'ready' if lines and supported and solution['solver_status'] == 'optimal' else 'best_effort')
    if solution['solver_status'] == 'limit_reached':
        warnings.append('Search node limit reached: the returned incumbent is not certified optimal; no infeasibility claim is made.')
    if candidates and not lines:
        warnings.append('No priced post was selected within the conservative budget in the explored search.')
    if conditional:
        warnings.append('Unpriced conditional candidates are retained separately and are not included as zero-cost commitments.')
    if any(item['status'] in {'proxy_unmet', 'not_evaluable'} for item in checks.values()):
        warnings.append('One or more client KPIs remain unmet or not evaluable under the stated planning assumptions.')
    if not configured_checks:
        warnings.append('No supported KPI targets were found; none were invented.')
    if exception:
        warnings.append('Out-of-corridor proposed pricing requires final human approval of the supplied exception reason.')
    previous = old_plan.get('revision', 0) if isinstance(old_plan, dict) else 0
    if type(previous) is not int or previous < 0:
        raise ValueError('Previous portfolio revision must be a nonnegative integer')
    assumptions = [
        'Historical medians times post quantities are planning proxies, not guaranteed future delivery or unique reach.',
        'Effects are not separated by image/video in the input; no video uplift is invented.',
        'An all-image example can result from equal effect proxies, image preference, and lower cost. Video-as-secondary has no invented fixed quota; explicit per-creator format restrictions are honored.',
        'The default one-post-per-creator cap is a development assumption. Explicit caps replace it for this revision.',
        'The +/-20% platform-reference corridor is a default pricing policy; exceptions retain a reason for final human review.',
        'Budget uses each proposed price range upper bound; lower negotiated prices are not presumed secured.',
        'Pricing uses sourced, matched industry observations when sufficient; internal peer quotes are explicitly labelled sample references, not verified market prices.',
        'Fees not quantified in the source budget have not been invented or silently added.',
    ]
    benchmark_kinds = {line['valuation']['status'] for line in lines}
    benchmark_status = (next(iter(benchmark_kinds)) if len(benchmark_kinds) == 1 else
                        'mixed' if benchmark_kinds else 'not_provided')
    if benchmark_status == 'platform_fallback':
        benchmark_status = 'not_provided'
    history = deepcopy(old_plan.get('revision_history') or [])
    if previous:
        history.append({key: deepcopy(old_plan.get(key)) for key in
                        ('revision', 'configuration', 'budget', 'feedback', 'feedback_id', 'input_fingerprint')})
    result = {
        'schema_version': '1.0', 'status': status, 'solver_status': solution['solver_status'],
        'search_nodes': solution['search_nodes'], 'max_search_nodes': max_nodes,
        'optimality_proven': solution['solver_status'] == 'optimal',
        'objective_order': ['prefer_all_evaluable_numeric_kpi_constraints_met',
                            'min_normalized_exposure_and_interaction_shortfall',
                            'min_normalized_aggregate_cpe_excess',
                            'max_upstream_score_once_per_creator', 'soft_format_preference',
                            'min_conservative_cost'],
        'lines': lines,
        'budget': {'currency': 'CNY', 'limit': _money(limit), 'source': budget_source,
                   'total_low': _money(totals['low']), 'total_suggested': _money(totals['suggested']),
                   'total_upper': _money(totals['high']), 'remaining_upper': _money(limit - totals['high']),
                   'within_budget': totals['high'] <= limit, 'basis': 'upper_bound'},
        'kpi_checks': checks, 'assumptions': assumptions, 'warnings': warnings,
        'conditional_candidates': conditional,
        'revision': previous + 1, 'feedback': _json_copy(feedback),
        'feedback_id': str(feedback_id) if feedback_id is not None else None,
        'revision_history': history, 'input_fingerprint': fingerprint,
        'configuration': _json_copy(cfg),
        'planning': planning,
        'pricing_policy': {'price_multiplier': float(multiplier),
                           'mode': 'explicit_multiplier' if 'price_multiplier' in cfg else 'benchmark_auto',
                           'exception_reason': reason.strip() if isinstance(reason, str) else None,
                           'requires_exception_approval': exception,
                           'industry_benchmark_status': benchmark_status},
        'industry_benchmark_status': benchmark_status,
        'post_caps': {'default': cfg.get('max_posts_per_creator', 1),
                      'per_creator': _json_copy(cfg.get('per_creator_max_posts') or {}),
                      'default_is_development_assumption': 'max_posts_per_creator' not in cfg},
        'approval_status': 'pending_final_human_review',
        'upstream_risks': {name: _json_copy(state.get(name, [] if name != 'audit_summary' else {}))
                           for name in ('audited_talents', 'audit_pending', 'supplement_results',
                                        'audit_unverified_checks', 'audit_summary')},
    }
    result['llm_annotations'] = {
        'status': ('generated_for_review' if planning.get('model_used') else
                   'unavailable' if planning.get('model_requested') else 'not_requested'),
        'execution_phase': 'before_solver', 'constraints': planning.get('accepted') or [],
        'explanation': planning.get('explanation'), 'explanation_verified': False,
    }
    return _json_copy(result)


class PortfolioAgent:
    """Source-grounded planner dispatching deterministic pricing and portfolio tools."""

    def __init__(self, llm_client: Any = None) -> None:
        self.llm_client = llm_client

    def build(self, state: dict, config: dict | None = None, *, feedback: Any = None) -> dict:
        return build_portfolio_plan(state, config, llm_client=self.llm_client, feedback=feedback)
