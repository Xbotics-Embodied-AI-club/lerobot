export const JOINT_ORDER = [
  'shoulder_pan',
  'shoulder_lift',
  'elbow_flex',
  'wrist_flex',
  'wrist_roll',
  'gripper',
];

export const JOINT_META = {
  shoulder_pan: { label: '肩部旋转', short: 'PAN' },
  shoulder_lift: { label: '肩部抬升', short: 'LIFT' },
  elbow_flex: { label: '肘部弯曲', short: 'ELBOW' },
  wrist_flex: { label: '腕部俯仰', short: 'W.FLEX' },
  wrist_roll: { label: '腕部旋转', short: 'W.ROLL' },
  gripper: { label: '夹爪', short: 'GRIP' },
};

export const STATUS_TEXT = {
  pending: '待标定',
  active: '进行中',
  pass: '已通过',
  error: '异常',
};

export const PHASE_TEXT = {
  middle: '中位确认',
  ranges: '限位记录',
  review: '结果复核',
  writing: '写入中',
  done: '已完成',
  error: '异常',
  cancelled: '已取消',
};

export const ACTION_TEXT = {
  confirm_middle: '确认中位',
  confirm_limit: '记录限位',
  write: '写入标定',
  retry_joint: '重试当前关节',
};

export function jointLabel(name, fallback) {
  return JOINT_META[name]?.label || fallback || name;
}
