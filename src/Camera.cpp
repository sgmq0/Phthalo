#include "stdafx.h"
#include "Camera.h"

Camera::Camera(XMFLOAT3 position, XMFLOAT3 up, float yaw, float pitch) :
    m_position(position),
    m_up(up),
    m_yaw(yaw),
    m_pitch(pitch)
{
    updateCameraVectors();
}

void Camera::updateCameraVectors() {
    XMFLOAT3 front;
    front.x = cos(XMConvertToRadians(m_yaw)) * cos(XMConvertToRadians(m_pitch));
    front.y = sin(XMConvertToRadians(m_pitch));
    front.z = sin(XMConvertToRadians(m_yaw)) * cos(XMConvertToRadians(m_pitch));

    XMVECTOR vFront = XMVector3Normalize(XMLoadFloat3(&front));
    XMStoreFloat3(&m_front, vFront);

    XMVECTOR vRight = XMVector3Normalize(XMVector3Cross(vFront, XMLoadFloat3(&m_up)));
    XMStoreFloat3(&m_right, vRight);

    XMVECTOR vUp = XMVector3Normalize(XMVector3Cross(vRight, vFront));
    XMStoreFloat3(&m_up, vUp);
}